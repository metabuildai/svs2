import glob
import os
import json
from scipy.io.wavfile import read
import pyworld
import numpy as np
import pandas as pd

root = '/root/shared/vvc'
pd_path = '/root/shared/pitch_check.csv'
df = pd.DataFrame(columns=['singer', 'song', 'start', 'end', 'lyric'])
data = {'singer' : list(), 'song' : list(), 'start': list(), 'end': list(), 'lyric': list()}

wav_list = glob.glob(os.path.join(root, 'S*', '*.wav'), recursive=True)

idx = 0
for path in wav_list:
    basename = os.path.basename(path).split('.')[0]
    dir_path = os.path.dirname(path)
    json_path = os.path.join(dir_path, '{}.json'.format(basename))

    singer = basename.split('_')[1]
    song = basename.split('_')[0]

    with open(json_path, 'r', encoding='UTF-8') as f:
        obj = json.load(f)
    notes = obj['Notes']

    sr, wav = read(path)

    f0, _ = pyworld.dio(wav.astype(float), sr, frame_period= 256.0 / sr * 1000)

    for note in notes:
        start = float(note['startTime'])
        end = float(note['endTime'])

        if len(note['Lyric']) == 0:
            continue

        sub = f0[int(start * sr / 256):int(end * sr / 256)]

        if np.all(sub == 0):
            print(singer, song, start, end, note['Lyric'])
            data['singer'].append(singer)
            data['song'].append(song)
            data['start'].append(start)
            data['end'].append(end)
            data['lyric'].append(note['Lyric'])

df = pd.DataFrame(data)
df.to_csv(pd_path, encoding='utf-8-sig')
