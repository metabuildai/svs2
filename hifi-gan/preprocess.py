import argparse
import tqdm
from preference import Config, load, save
import os
import pickle
from scipy.io.wavfile import read
import json
import math
import logging
import sys
import random


def parse_filepath(path: str):
    path_list = []
    with open(path) as f:
        path_list = f.readlines()
        # remove '\n'
        path_list = [path.split('\n')[0] for path in path_list]
    return path_list


def process(wave_file, desc, cf:Config, out_file, logger):
    save_file = list()

    for file in tqdm.tqdm(wave_file, desc=desc):
        logger.info('preprocess ' + file)

        basename = os.path.basename(file).split('.')[0]

        dir_path = os.path.dirname(file)
        mid_path = os.path.join(dir_path, '{}.json'.format(basename))

        with open(mid_path, 'r', encoding='UTF-8') as f:
            obj = json.load(f)
        notes = obj['Notes']

        segment_duration = cf.audio_segment_size / float(cf.sampling_rate)

        sampling_rate, audio = read(file)
        audio = audio / cf.max_wave_value

        segment_start = 0
        segment_end = 0
        for note in notes:
            if len(note['Lyric']) == 0:
                continue

            if segment_start == 0 and segment_end == 0:
                start = float(note['startTime'])
                end = float(note['endTime'])

                segment_start = start
                segment_end = end
            else:
                start = float(note['startTime'])
                end = float(note['endTime'])

                if segment_end + segment_duration > start:
                    segment_end = end
                else:
                    start_idx = math.floor(segment_start * cf.sampling_rate)
                    end_dix = math.ceil(segment_end * cf.sampling_rate)
                    wave = audio[start_idx:end_dix]

                    save_path = os.path.join(cf.preprocessed_dir, '{}_{}.pkl'.format(basename, start_idx))
                    with open(save_path, 'wb') as f:
                        pickle.dump(
                            {'audio': wave.tolist()},
                            f
                        )
                    save_file.append(save_path)

                    segment_start = start
                    segment_end = end
        start_idx = math.floor(segment_start * cf.sampling_rate)
        end_dix = math.ceil(segment_end * cf.sampling_rate)
        wave = audio[start_idx:end_dix]

        save_path = os.path.join(cf.preprocessed_dir, '{}_{}.pkl'.format(basename, start_idx))
        with open(save_path, 'wb') as f:
            pickle.dump(
                {'audio': wave.tolist()},
                f
            )
        save_file.append(save_path)

    random.shuffle(save_file)
    train_size = int(len(save_file) * 0.9)
    config.train_file = save_file[:train_size]
    config.valid_file = save_file[train_size:]


if __name__ == '__main__':
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
    parser.add_argument('--output_dir', '-o', help='save directory for preprocessed file')
    parser.add_argument('--config', help='preprocess configuration file')

    # argument parsing
    args = parser.parse_args()
    config_file = args.config
    file = args.file
    output_dir = args.output_dir

    # configuration setting
    if config_file is not None:
        config = load(config_file)
    else:
        config = Config()
    if output_dir is not None:
        config.preprocessed_dir = output_dir
    else:
        output_dir = config.preprocessed_dir
    config.wave_file = []
    config.train_file = []
    config.valid_file = []

    # parsing train wave file
    config.wave_file = parse_filepath(file)

    # make preprocess result folder
    os.makedirs(output_dir, exist_ok=True)
    print('preprocess directory: ' + output_dir)

    process(config.wave_file, 'wave preprocess', config, config.train_file, logger)

    path = os.path.join(output_dir, 'config.json')
    save(config, path)
