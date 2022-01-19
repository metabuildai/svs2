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
import json
import copy
import librosa
import pyworld
from audio.stft import MelSpectrogram
from utils import get_alignment, remove_outlier, average_by_duration
import math

MAX_WAV_VALUE = 32768.0


def text_to_id(text):
    id = list()
    for t in text:
        if t in ['sil']:
            id.append(1)
        elif t in ['sp']:
            id.append(2)
        elif ord('가') <= ord(t) <= ord('힣'):
            id.append(ord(t) - ord('가') + 3)
        else:
            print(t)
            exit()
    return id


def silence_alignment(notes):
    for note in notes:
        note['startTime'] = float(note['startTime'])
        note['endTime'] = float(note['endTime'])

        if len(note['Lyric']) == 0:
            note['Lyric'] = 'sil'

    # merge note
    pre_note = None
    for note in notes:
        start = note['startTime']
        end = note['endTime']
        if pre_note is None:
            pre_note = note
            pre_end = end
        else:
            mid = (pre_end + start) / 2

            pre_note['endTime'] = mid
            note['startTime'] = mid

            pre_note = note
            pre_end = end

    sil_note = list()
    for note in notes:
        if note['Lyric'] in ['sil']:
            sil_note.append(note)

    return sil_note


def alignment(notes, sr, hope_size):
    # time format transform (string to float)
    # time format transform (string to float)
    for note in notes:
        note['startTime'] = float(note['startTime'])
        note['endTime'] = float(note['endTime'])

        if len(note['Lyric']) == 0:
            note['Lyric'] = 'sil'

    # remove minimal hop
    new_notes = list()
    min_hop = hope_size / sr
    for note in notes:
        if (note['endTime'] - note['startTime']) >= min_hop:
            new_notes.append(note)
    notes = new_notes

    # check subset of pre note
    pre_note = None
    new_notes = list()
    for note in notes:
        start = note['startTime']
        end = note['endTime']
        w = note['Lyric']
        if pre_note is not None:
            if pre_start <= start and end <= pre_end:
                if pre_w in ['sil'] and w not in ['sil']:
                    new_notes[-1]['endTime'] = start
                elif pre_w not in ['sil'] and w in ['sil']:
                    continue
                else:
                    new_notes[-1]['endTime'] = start

        new_notes.append(copy.deepcopy(note))
        pre_note = note
        pre_start = start
        pre_end = end
        pre_w = w
    notes = new_notes

    # check overlap note
    pre_note = None
    new_notes = list()
    for note in notes:
        start = note['startTime']
        end = note['endTime']
        w = note['Lyric']
        if pre_note is not None:
            if start < pre_end:
                if pre_w in ['sil'] and w not in ['sil']:
                    new_notes[-1]['endTime'] = start
                elif pre_w not in ['sil'] and w in ['sil']:
                    note['startTime'] = pre_end
                else:
                    new_notes[-1]['endTime'] = start

        new_notes.append(copy.deepcopy(note))
        pre_note = note
        pre_start = start
        pre_end = end
        pre_w = w
    notes = new_notes

    # remove minimal hop
    new_notes = list()
    min_hop = hope_size / sr
    for note in notes:
        if (note['endTime'] - note['startTime']) >= min_hop:
            new_notes.append(note)
    notes = new_notes

    # interval less than min_hop
    pre_note = None
    new_notes = list()
    for note in notes:
        start = note['startTime']
        end = note['endTime']
        w = note['Lyric']
        if pre_note is not None:
            if (start - pre_end) < min_hop:
                mid = (start + pre_end) / 2
                new_notes[-1]['endTime'] = mid
                note['startTime'] = mid
        new_notes.append(copy.deepcopy(note))
        pre_note = note
        pre_start = start
        pre_end = end
        pre_w = w
    notes = new_notes

    min_hop = hope_size / sr
    for note in notes:
        start = note['startTime']
        end = note['endTime']

        if (end - start) < min_hop:
            print('not min hop size', note)
            pad = (min_hop - (end - start)) / 2
            note['startTime'] = start - pad
            note['endTime'] = end + pad
            print('after: ', note)

    # check start note
    start_note = notes[0]
    for i, note in enumerate(notes):
        if note['Lyric'] not in ['sil', 'sp']:
            start_note = note
            break
    start_time = start_note['startTime']

    # convert hop position
    for note in notes:
        note['startHop'] = round((note['startTime'] - start_time) * sr / hope_size)
        note['endHop'] = round((note['endTime'] - start_time) * sr / hope_size)

    # black to sp note
    new_notes = list()
    pre_note = None
    for note in notes:
        start = note['startHop']
        if pre_note is not None:
            pre_end = pre_note['endHop']
            if start != pre_end:
                n = copy.deepcopy(note)
                n['startHop'] = pre_end
                n['endHop'] = start
                n['Lyric'] = 'sp'
                n['midiNum'] = 1
                new_notes.append(n)
        pre_note = note
        new_notes.append(copy.deepcopy(note))
    notes = new_notes

    new_notes = list()
    for note in notes:
        for hop in range(note['startHop'], note['endHop']):
            copy_note = copy.deepcopy(note)
            copy_note['startHop'] = hop
            copy_note['endHop'] = hop + 1
            new_notes.append(copy_note)
    notes = new_notes

    pre_note = None
    for note in notes:
        if pre_note is not None:
            if pre_note['endHop'] != note['startHop']:
                print('not continuous pre: ', pre_note)
                print('not continuous cur', note)
                exit()
        pre_note = note

    # check start voice
    start_list = list()
    for i, note in enumerate(notes):
        if note['Lyric'] not in ['sil', 'sp']:
            start_list.append(i)
            break

    # sentence voice
    sentence_list = list()
    for start_idx in start_list:
        sentence = list()
        start = notes[start_idx]['startHop']
        sentence.append(copy.deepcopy(notes[start_idx]))
        for idx in range(start_idx + 1, len(notes)):
            sentence.append(copy.deepcopy(notes[idx]))
        sentence_list.append(sentence)

    alignment_list = list()
    for sentence in sentence_list:
        start = (sentence[0]['startHop'] * hope_size) / sr + start_time
        end = (sentence[-1]['endHop'] * hope_size) / sr + start_time
        text = list()
        duration = list()
        pitch = list()
        for note in sentence:
            text.append(note['Lyric'])
            pitch.append(note['midiNum'])
            duration.append(1)
        alignment_list.append({'text': text, 'duration': duration, 'pitch': pitch, 'start': start, 'end': end})

    return alignment_list



def inference(files, model, vocoder, mel_spectogram, output_dir, config, desc):
    for file in tqdm.tqdm(files, desc=desc):
        basename = os.path.splitext(os.path.basename(file))[0]

        txt_grid_dir = os.path.dirname(file)
        txt_grid = os.path.join(txt_grid_dir, '{}.json'.format(basename))
        with open(txt_grid, 'r', encoding='UTF-8') as f:
            obj = json.load(f)
        notes = obj['Notes']

        align = alignment(notes, config.sampling_rate, config.hop_length)

        audio, sampling_rate = librosa.load(file, sr=config.sampling_rate)
        out_audio = audio.copy()
        out_audio = out_audio * MAX_WAV_VALUE
        out_audio = out_audio.astype('int16')
        out_audio[:] = 0

        for i, ali in enumerate(align):
            text = ali['text']
            duration = ali['duration']
            start = ali['start']
            end = ali['end']

            text_id = text_to_id(text)

            wav = audio[int(config.sampling_rate * start):int(config.sampling_rate * end)]

            f0, _ = pyworld.dio(wav.astype(float), config.sampling_rate,
                                frame_period=config.hop_length / config.sampling_rate * 1000)
            f0 = f0[:sum(duration)]

            min = np.min([len(text_id), f0.shape[0]])
            text_id = text_id[:min]
            f0 = f0[:min]

            text = np.array(text_id)
            src_len = np.array(text.shape[0])
            phone_mask = np.full(src_len, False)
            duration = np.array([duration])
            f0 = np.array([f0])

            text = torch.from_numpy(text.reshape(1, -1)).to(device)
            src_len = torch.from_numpy(src_len.reshape(1, -1)).to(device)
            phone_mask = torch.from_numpy(phone_mask.reshape(1, -1)).to(device)
            duration = torch.from_numpy(duration).long().to(device)
            f0 = torch.from_numpy(f0).long().to(device)

            with torch.no_grad():
                mel_postnet = model.inference(text, duration, f0, f0, phone_mask)

            mel = mel_scaler.inverse_transform(mel_postnet.squeeze().cpu().detach())
            mel_trans = torch.from_numpy(mel).transpose(0, 1).to(device)

            wave = vocoder(mel_trans.unsqueeze(0).cpu())
            wave = wave * MAX_WAV_VALUE
            wave = wave.cpu().detach().numpy().astype('int16')

            margin = (wave.shape[2] - (int(config.sampling_rate * end) - int(config.sampling_rate * start))) // 2
            if margin <= 0:
                wave = wave[0, 0, :]
            else:
                wave = wave[0, 0, margin:-margin]
            wave_start = int(config.sampling_rate * start)
            out_audio[wave_start:wave_start+wave.shape[0]] = wave
        path = os.path.join(output_dir, basename + '.wav')
        write(path, config.sampling_rate, out_audio)


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
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = False
    if torch.cuda.is_available() and config.gpu_index >= 0:
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
    print('train directory : ', os.path.join(output_dir, 'traioin'))
    print('valid directory : ', os.path.join(output_dir, 'valid'))

    if sampling_num is not None:
        if sampling_num < len(config.wave_file):
            wave_file = config.wave_file.copy()
            random.shuffle(wave_file)
            wave_file = wave_file[:sampling_num]
        else:
            wave_file = config.wave_file
    else:
        wave_file = config.wave_file
    print('inference file: ', len(wave_file))

    print('{s:{c}^{n}}\n'.format(s='complete: setup step', n=50, c='-'))

    # dataset
    mel_scaler, f0_scaler, energy_scaler = load_scaler(config.mel_scaler, config.f0_scaler, config.energy_scaler)

    # model
    model = FastSpeech2(
        config.decoder_hidden, config.n_mel_channels, ord('힣') - ord('가') + 3, config.max_seq_len, config.encoder_hidden,
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
    vocoder.to(torch.device('cpu')).eval()

    print('{s:{c}^{n}}\n'.format(s='complete: model step', n=50, c='-'))

    mel_spectogram = MelSpectrogram(config.sampling_rate, config.filter_length, config.n_mel_channels,
                                    config.win_length,
                                    config.hop_length, config.mel_fmin, config.mel_fmax)

    inference(wave_file, model, vocoder, mel_spectogram, output_dir, config, 'inference dataset')
