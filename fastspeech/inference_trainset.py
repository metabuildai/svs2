import argparse
import numpy as np
import preference
import torch
import os
import shutil
from fastspeech2 import FastSpeech2
from vocoder.generator import Generator
import random
import tqdm
import pickle
from scaler import load_scaler
from text.korean import symbols
from scipy.io.wavfile import write

MAX_WAV_VALUE = 32768.0

if __name__ == "__main__":
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='JSON file for configuration')
    parser.add_argument('--dir', '-d', default='generated_file', help='directory for saving generated file')
    parser.add_argument('--file', '-f', help='checkpoint file to load generator model')
    parser.add_argument('--vocoder', '-v', help='vocoder checkpoint file')
    parser.add_argument('--sample', '-s', type=int, help='sampling number for inference train/valid data')
    parser.add_argument('--remove', '-r', action='store_true', help='remove previous generated file')

    # argument parsing
    args = parser.parse_args()
    config_file = args.config
    output_dir = args.dir
    checkpoint_file = args.file
    vocoder_file = args.vocoder
    sampling_num = args.sample
    remove = args.remove

    # configuration setting
    config = preference.load(config_file)
    if checkpoint_file is None:
        checkpoint_file = config.last_checkpoint_file

    # setup
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = '3'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available() and config.gpu_index >= 0:
        # device = torch.device('cuda:{:d}'.format(config.gpu_index))
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    os.makedirs(output_dir, exist_ok=True)
    if remove is True:
        if os.path.isdir(os.path.join(output_dir, 'train')) is True:
            shutil.rmtree(os.path.join(output_dir, 'train'))
        if os.path.isdir(os.path.join(output_dir, 'valid')) is True:
            shutil.rmtree(os.path.join(output_dir, 'valid'))
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=True)
    os.makedirs(os.path.join(output_dir, 'valid'), exist_ok=True)
    print('train directory : ', os.path.join(output_dir, 'train'))
    print('valid directory : ', os.path.join(output_dir, 'valid'))

    if sampling_num is not None:
        if sampling_num > len(config.train_file) or sampling_num > len(config.valid_file):
            print('sampling number is over')
            exit()
        else:
            train_file = config.train_file.copy()
            random.shuffle(train_file)
            train_file = train_file[:sampling_num]

            valid_file = config.valid_file.copy()
            random.shuffle(valid_file)
            valid_file = valid_file[:sampling_num]
    else:
        train_file = config.train_file
        valid_file = config.valid_file
    print('inference train file: ', len(train_file))
    print('inference valid file: ', len(valid_file))

    print('{s:{c}^{n}}\n'.format(s='complete: setup step', n=50, c='-'))

    # dataset
    mel_scaler, f0_scaler, energy_scaler = load_scaler(config.mel_scaler, config.f0_scaler, config.energy_scaler)

    # model
    model = FastSpeech2(
        config.decoder_hidden, config.n_mel_channels, ord('힣') - ord('가') + 2, config.max_seq_len, config.encoder_hidden,
        config.encoder_head, config.decoder_head, config.decoder_dropout, config.fft_filter_size,
        config.fft_kernel_size, config.encoder_dropout, config.log_offset, config.variance_predictor_filter_size,
        config.variance_predictor_kernel_size, config.variance_predictor_dropout
    )
    model.load_state_dict(torch.load(checkpoint_file)['model'])
    model = model.to(device)
    model.eval()

    vocoder = Generator(
        resblock_kernel_size=config.resblock_kernel_size,
        upsample_rate=config.upsample_rate,
        upsample_kernel_size=config.upsample_kernel_size,
        upsample_initial_channel=config.upsample_initial_channel,
        resblock_dilation_size=config.resblock_dilation_size,
        num_mel=config.n_mel_channels
    )
    dict = torch.load(vocoder_file)
    vocoder.load_state_dict(torch.load(vocoder_file)['generator'])
    vocoder.to(device).eval()

    print('{s:{c}^{n}}\n'.format(s='complete: model step', n=50, c='-'))

    # inference train set
    for file in tqdm.tqdm(train_file, desc='inference train dataset'):
        basename = os.path.splitext(os.path.basename(file))[0]
        with open(file, 'rb') as f:
            data = pickle.load(f)
            text = data['text_id']
            duration = data['duration']
            f0 = data['f0']
            std_f0 = f0_scaler.transform(np.array(data['f0']).reshape(-1, 1)).flatten()
        text = np.array(text)
        src_len = np.array(text.shape[0])
        phone_mask = np.full(src_len, False)
        duration = np.array([duration])
        f0 = np.array([f0])
        std_f0 = np.array([std_f0])

        text = torch.from_numpy(text.reshape(1, -1)).to(device)
        src_len = torch.from_numpy(src_len.reshape(1, -1)).to(device)
        phone_mask = torch.from_numpy(phone_mask.reshape(1, -1)).to(device)
        duration = torch.from_numpy(duration).long().to(device)
        f0 = torch.from_numpy(f0).long().to(device)
        std_f0 = torch.from_numpy(std_f0).float().to(device)

        with torch.no_grad():
            mel_postnet = model.inference(text, duration, f0, std_f0, phone_mask)

        mel = mel_scaler.inverse_transform(mel_postnet.squeeze().cpu().detach())
        mel_trans = torch.from_numpy(mel).transpose(0, 1).to(device)

        wave = vocoder(mel_trans.unsqueeze(0))
        wave = wave * MAX_WAV_VALUE
        audio = wave.cpu().detach().numpy().astype('int16')
        path = os.path.join(output_dir, 'train', basename + '.wav')
        write(path, config.sampling_rate, audio)

    # inference valid set
    for file in tqdm.tqdm(valid_file, desc='inference valid dataset'):
        basename = os.path.splitext(os.path.basename(file))[0]
        with open(file, 'rb') as f:
            data = pickle.load(f)
            text = data['text_id']
            duration = data['duration']
            f0 = data['f0']
            std_f0 = f0_scaler.transform(np.array(data['f0']).reshape(-1, 1)).flatten()
        text = np.array(text)
        src_len = np.array(text.shape[0])
        phone_mask = np.full(src_len, False)
        duration = np.array([duration])
        f0 = np.array([f0])
        std_f0 = np.array([std_f0])

        text = torch.from_numpy(text.reshape(1, -1)).to(device)
        src_len = torch.from_numpy(src_len.reshape(1, -1)).to(device)
        phone_mask = torch.from_numpy(phone_mask.reshape(1, -1)).to(device)
        duration = torch.from_numpy(duration).long().to(device)
        f0 = torch.from_numpy(f0).long().to(device)
        std_f0 = torch.from_numpy(std_f0).float().to(device)

        with torch.no_grad():
            mel_postnet = model.inference(text, duration, f0, std_f0, src_mask=phone_mask)

        mel = mel_scaler.inverse_transform(mel_postnet.squeeze().cpu().detach())
        mel_trans = torch.from_numpy(mel).transpose(0, 1).to(device)

        wave = vocoder(mel_trans.unsqueeze(0))
        wave = wave * MAX_WAV_VALUE
        audio = wave.cpu().detach().numpy().astype('int16')
        path = os.path.join(output_dir, 'valid', basename + '.wav')
        write(path, config.sampling_rate, audio)