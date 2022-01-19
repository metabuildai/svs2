from sklearn.preprocessing import StandardScaler
from os.path import join
import joblib


def build_scaler():
    mel_scaler = StandardScaler(copy=False)
    f0_scaler = StandardScaler(copy=False)
    energy_scaler = StandardScaler(copy=False)

    return mel_scaler, f0_scaler, energy_scaler


def load_scaler(mel_path, f0_path, energy_path):
    with open(mel_path, 'rb') as f:
        mel_scaler = joblib.load(f)
        print('mel spectrogram scaler file: ', mel_path)

    with open(f0_path, 'rb') as f:
        f0_scaler = joblib.load(f)
        print('f0 scaler file: ', f0_path)

    with open(energy_path, 'rb') as f:
        energy_scaler = joblib.load(f)
        print('energy scaler file: ', energy_path)

    return mel_scaler, f0_scaler, energy_scaler


def save_scaler(mel, f0, energy, dir):
    mel_path = join(dir, 'mel_scaler.bin')
    with open(mel_path, 'wb') as f:
        joblib.dump(mel, f)

    f0_path = join(dir, 'f0_scaler.bin')
    with open(f0_path, 'wb') as f:
        joblib.dump(f0, f)

    energy_path = join(dir, 'energy_scaler.bin')
    with open(energy_path, 'wb') as f:
        joblib.dump(energy, f)

    return mel_path, f0_path, energy_path
