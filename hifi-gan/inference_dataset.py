import argparse
import torch
import os
import shutil
import random
import tqdm
from utils import MelSpectrogram
from train import build_model
from preference import Config, load
from scipy.io.wavfile import write, read

MAX_WAV_VALUE = 32768.0


def load_model(path: str, cf: Config, device):
    generator, _, _, _, _, _, _ = build_model(cf, device)

    load_state = torch.load(path)

    generator.load_state_dict(load_state['generator'])
    generator = generator.to(device)

    return generator


def inference(generator, mel_spectrogram: MelSpectrogram, files, cf: Config, dir, desc):
    for file in tqdm.tqdm(files, desc=desc):
        basename = os.path.splitext(os.path.basename(file))[0]

        sampling_rate, audio = read(file)
        audio = audio / cf.max_wave_value
        wav = torch.FloatTensor(audio).to(device)

        x = mel_spectrogram(wav.unsqueeze(0))
        try:
            with torch.no_grad():
                fake_wav = generator(x)
        except:
            print('error file: ', file)
        audio = fake_wav.squeeze()
        audio = audio * MAX_WAV_VALUE
        audio = audio.cpu().detach().numpy().astype('int16')

        output_file = os.path.join(dir, basename + '.wav')
        write(output_file, config.sampling_rate, audio)


if __name__ == "__main__":
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='JSON file for configuration')
    parser.add_argument('--dir', '-d', default='generated_file', help='directory for saving generated file')
    parser.add_argument('--file', '-f', help='checkpoint file to load generator model')
    parser.add_argument('--sample', '-s', type=int, help='sampling number for inference train/valid data')
    parser.add_argument('--remove', '-r', action='store_true', help='remove previous generated file')
    parser.add_argument('--gpu', '-g', type=int, help='gpu device index')


    # argument parsing
    args = parser.parse_args()
    config_file = args.config
    output_dir = args.dir
    checkpoint_file = args.file
    sampling_num = args.sample
    remove = args.remove
    gpu = args.gpu

    # configuration setting
    config = load(config_file)
    if checkpoint_file is None:
        checkpoint_file = config.last_checkpoint_file
    if gpu is not None:
        config.gpu_index = gpu

    # setup
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available() and config.gpu_index >= 0:
        device = torch.device('cuda:{:d}'.format(config.gpu_index))
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
        train_file = config.train_wave_file.copy()
        if len(train_file) > sampling_num:
            random.shuffle(train_file)
            train_file = train_file[:sampling_num]

        valid_file = config.valid_wave_file.copy()
        if len(valid_file) > sampling_num:
            random.shuffle(valid_file)
            valid_file = valid_file[:sampling_num]
    else:
        train_file = config.train_wave_file
        valid_file = config.valid_wave_file
    print('inference train file: ', len(train_file))
    print('inference valid file: ', len(valid_file))

    print('{s:{c}^{n}}\n'.format(s='complete: setup step', n=50, c='-'))

    # model
    generator = load_model(checkpoint_file, config, device)
    generator.eval()

    mel_spectrogram = MelSpectrogram(config.sampling_rate, config.n_fft, config.num_mel, config.win_size, config.hop_size)
    mel_spectrogram = mel_spectrogram.to(device)

    # generator.eval()
    inference(generator, mel_spectrogram, train_file, config, os.path.join(output_dir, 'train'), 'inference train dataset')
    inference(generator, mel_spectrogram, valid_file, config, os.path.join(output_dir, 'valid'), 'inference valid dataset')
