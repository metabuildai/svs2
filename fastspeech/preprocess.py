import os
import pickle
import tqdm
import argparse
from preference import Config, load, save
from utils import get_alignment, remove_outlier, average_by_duration
import librosa
import pyworld
import torch
from audio.stft import MelSpectrogram
from text.korean import text_to_sequence
from scaler import build_scaler, save_scaler
from librosa import filters
import copy
from jamo import h2j, j2hcj
from g2pk import G2p
import json
import numpy as np
import random
from scipy.io.wavfile import read
import math
import logging
import sys
import random


silence = ['sp', 'sil']


def parse_filepath(path: str):
    path_list = []
    with open(path) as f:
        path_list = f.readlines()
        # remove '\n'
        path_list = [path.split('\n')[0] for path in path_list]
    return path_list


def alignment(notes, sr, hope_size, segment):
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

    for i, note in enumerate(notes):
        start = note['startHop']
        lyric = note['Lyric']

        if lyric in ['sil', 'sp']:
            continue
        elif (start - notes[start_list[-1]]['startHop']) > (3.0 * sr / hope_size):
            start_list.append(i)

    # sentence voice
    sentence_list = list()
    for start_idx in start_list:
        sentence = list()
        start = notes[start_idx]['startHop']
        sentence.append(copy.deepcopy(notes[start_idx]))
        for idx in range(start_idx + 1, len(notes)):
            end = notes[idx]['endHop']
            if (end - start) <= (6.0 * sr / hope_size):
                sentence.append(copy.deepcopy(notes[idx]))
            else:
                break
        if sentence[-1]['endHop'] - sentence[0]['startHop'] == math.floor(6.0 * sr / hope_size):
            sentence_list.append(sentence)

    alignment_list = list()
    for sentence in sentence_list:
        if sentence[-1]['endHop'] - sentence[0]['startHop'] != 1033:
            print('diference hop', sentence[-1]['endHop'] - sentence[0]['startHop'])
            continue
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


def process(wave_file, textgrid_file, mel_scaler, f0_scaler, energy_scaler, config: Config, desc: str, logger):
    mel_spectogram = MelSpectrogram(config.sampling_rate, config.filter_length, config.n_mel_channels, config.win_length,
                                    config.hop_length, config.mel_fmin, config.mel_fmax)

    save_file = list()

    for i in tqdm.tqdm(range(len(wave_file)), desc=desc):
        basename = os.path.basename(wave_file[i]).split('.')[0]
        logger.info('preprocess ' + basename)

        with open(textgrid_file[i], 'r', encoding='UTF-8') as f:
            obj = json.load(f)
        notes = obj['Notes']
        align_list = alignment(notes, config.sampling_rate, config.hop_length, None)

        audio, sampling_rate = librosa.load(wave_file[i], sr=config.sampling_rate)
        sr, data = read(wave_file[i])

        if data.dtype != np.int16:
            print('audio format not int16: ', basename)
            continue

        for i, align in enumerate(align_list):
            text = align['text']
            duration = align['duration']
            start = align['start']
            end = align['end']

            text_id = text_to_id(text)

            wav = audio[int(config.sampling_rate * start):math.ceil(config.sampling_rate * end)]

            # fundamental frequency
            f0, _ = pyworld.dio(wav.astype(float), config.sampling_rate, frame_period=config.hop_length / config.sampling_rate * 1000)
            f0 = f0[:sum(duration)]

            if np.all(f0 == 0):
                continue

            mel = mel_spectogram(torch.tensor(wav).unsqueeze(0))
            mel = mel.squeeze(0).numpy().astype(float)[:, :sum(duration)]

            # f0 = average_by_duration(f0, np.array(duration))
            if mel.shape[1] != len(duration):
                print(i)
                continue

            mel_scaler.partial_fit(mel.T)
            f0_scaler.partial_fit(f0[f0 != 0].reshape(-1, 1))

            config.max_duration = max(config.max_duration, max(duration))
            config.max_pitch = max(config.max_pitch, int(max(f0)))

            file = os.path.join(config.preprocess_dir, '{}_{}.pkl'.format(basename, i))
            with open(file, 'wb') as f:
                pickle.dump(
                    {'text': text,
                     'text_id': text_id,
                     'duration': duration,
                     'f0': f0.tolist(),
                     'mel': mel.tolist()},
                    f
                )
                save_file.append(file)

    random.shuffle(save_file)
    train_size = int(len(save_file) * 0.9)
    config.train_file = save_file[:train_size]
    config.valid_file = save_file[train_size:]


def match_textgrid(wave_file, dir):
    text_file = list()
    for file in wave_file:
        basename = os.path.basename(file).split('.')[0]
        text_file.append(os.path.join(dir, '{}.json'.format(basename)))
    return text_file

    
if __name__ == "__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(message)s', '%Y-%m-%d %H:%M:%S %Z%z')

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler('log.txt')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    cmd = ' '.join(sys.argv)
    logger.info('python ' + cmd)

    print('Initializing PreProcess..')

    parser = argparse.ArgumentParser()
    parser.add_argument('file', help='file that have train audio file path')
    parser.add_argument('textgrid_dir', help='directory that have textgrid file with audio same name')
    parser.add_argument('--output_dir', '-o', help='save directory for preprocessed file')
    parser.add_argument('--config', '-c', help='preprocess configuration file')

    # argument parsing
    args = parser.parse_args()
    config_file = args.config
    textgrid_dir = args.textgrid_dir
    file = args.file
    output_dir = args.output_dir

    # configuration setting
    if config_file is not None:
        config = load(config_file)
    else:
        config = Config()
    if output_dir is not None:
        config.preprocess_dir = output_dir
    config.wave_file = []
    config.textgrid_file = []
    config.train_file = []
    config.valid_file = []

    # parsing train wave file
    config.wave_file = parse_filepath(file)
    config.textgrid_file = match_textgrid(config.wave_file, textgrid_dir)

    # make preprocess result folder
    os.makedirs(output_dir, exist_ok=True)
    print('make preprocess directory: ', output_dir)

    mel_scaler, f0_scaler, energy_scaler = build_scaler()

    process(config.wave_file, config.textgrid_file, mel_scaler, f0_scaler, energy_scaler, config, 'wave preprocess', logger)

    mel_path, f0_path, energy_path = save_scaler(mel_scaler, f0_scaler, energy_scaler, config.preprocess_dir)
    config.mel_scaler = mel_path
    config.f0_scaler = f0_path
    config.energy_scaler = energy_path

    path = os.path.join(output_dir, 'config.json')
    save(config, path)
