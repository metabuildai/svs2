import argparse
import re
from string import punctuation
import numpy as np
from g2pk import G2p
from jamo import h2j
import preference
import torch
from fastspeech2 import FastSpeech2
from vocoder.vocgan_generator import Generator
import utils
from scaler import load_scaler
from text.korean import symbols, text_to_sequence


def kor_preprocess(text):
    text = text.rstrip(punctuation)

    g2p=G2p()
    phone = g2p(text)
    print('after g2p: ',phone)
    phone = h2j(phone)
    print('after h2j: ',phone)
    phone = list(filter(lambda p: p != ' ', phone))
    phone = '{' + '}{'.join(phone) + '}'
    print('phone: ',phone)
    phone = re.sub(r'\{[^\w\s]?\}', '{sp}', phone)
    print('after re.sub: ',phone)
    phone = phone.replace('}{', ' ')

    print('|' + phone + '|')
    sequence = np.array(text_to_sequence(phone))
    sequence = np.stack([sequence])
    return torch.from_numpy(sequence).long()


if __name__ == "__main__":
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('config', help='JSON file for configuration')
    parser.add_argument('--file', '-f', help='checkpoint file to load generator model')
    parser.add_argument('--hifi-gan', '-v', help='hifi-gan checkpoint file')
    parser.add_argument('--out', '-o', default='generated.wav', help='generated file')

    # argument parsing
    args = parser.parse_args()
    config_file = args.config
    checkpoint_file = args.file
    vocoder_file = args.vocoder
    save_file = args.out

    # configuration setting
    config = preference.load(config_file)
    if checkpoint_file is None:
        checkpoint_file = config.last_checkpoint_file

    # setup
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available() and config.gpu_index >= 0:
        # device = torch.device('cuda:{:d}'.format(config.gpu_index))
        device = torch.device('cuda'.format(config.gpu_index))
    else:
        device = torch.device('cpu')

    # os.makedirs(output_dir, exist_ok=True)

    print('{s:{c}^{n}}\n'.format(s='complete: setup step', n=50, c='-'))

    # dataset
    mel_scaler, f0_scaler, energy_scaler = load_scaler(config.mel_scaler, config.f0_scaler, config.energy_scaler)

    # model
    model = FastSpeech2(
        config.decoder_hidden, config.n_mel_channels, len(symbols) + 1, config.max_seq_len, config.encoder_hidden,
        config.encoder_head, config.decoder_head, config.decoder_dropout, config.fft_filter_size,
        config.fft_kernel_size, config.encoder_dropout, config.log_offset, config.variance_predictor_filter_size,
        config.variance_predictor_kernel_size, config.variance_predictor_dropout
    )
    model.load_state_dict(torch.load(checkpoint_file)['model'])
    model = model.to(device)
    model.requires_grad = False
    model.eval()

    vocoder = Generator(config.n_mel_channels, 4, ratios=[4, 4, 2, 2, 2, 2], mult=256, out_band=1)
    vocoder.load_state_dict(torch.load(vocoder_file)['model_g'])
    vocoder.to(device).eval()

    print('{s:{c}^{n}}\n'.format(s='complete: model step', n=50, c='-'))

    # inference text
    text = '안녕하세요 메타빌드 에이아이 연구실입니다'
    phone = kor_preprocess(text)
    src_len = np.array(phone.shape[0])
    phone_mask = np.full(src_len, False)

    phone = torch.from_numpy(phone.reshape(1, -1)).to(device)
    src_len = torch.from_numpy(src_len.reshape(1, -1)).to(device)
    phone_mask = torch.from_numpy(phone_mask.reshape(1, -1)).to(device)

    mel, mel_postnet, log_duration_output, f0_output, energy_output, _, _, mel_len = model(phone, src_len, src_mask=phone_mask)

    mel = mel_scaler.inverse_transform(mel_postnet.squeeze().cpu().detach())
    mel_trans = torch.from_numpy(mel).transpose(0, 1).to(device)

    utils.vocgan_infer(mel_trans, vocoder, path=save_file)
