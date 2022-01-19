import torch
from torch.utils import data
import numpy as np


def pad_1d(array, width):
    return np.stack([np.pad(x, (0, width - x.shape[0])) for x in array])


def pad_2d(array, width):
    return np.stack([np.pad(x, ((0, width - x.shape[0]), (0, 0))) for x in array])


def length_mask(length, max):
    ids = np.arange(0, max).reshape(1, -1).repeat(length.shape[0], axis=0)

    return ids >= length.reshape(-1, 1).repeat(max, axis=1)

# def silence_mask(text, duration):
#
#
#     return sil_mask
def silence_matrix(text, duration):
    batch = len(text)
    s = list()
    for i in range(batch):
        m = np.concatenate([np.ones(d) * text[i][j] for j, d in enumerate(duration[i])])
        s.append(m)
    return s


class Dataset(data.Dataset):
    def __init__(self, txt_id_list, mel_list, duration_list, f0_list, std_f0_list, mel_scaler, f0_scaler, energy_scaler, log_offset):
        self.txt_id_list = txt_id_list
        self.mel_list = mel_list
        self.duration_list = duration_list
        self.f0_list = f0_list
        self.std_f0_list = std_f0_list
        self.mel_scaler = mel_scaler
        self.f0_scaler = f0_scaler
        self.energy_scaler = energy_scaler
        self.log_offset = log_offset

    def __len__(self):
        return len(self.txt_id_list)

    def __getitem__(self, idx):
        return {'text': self.txt_id_list[idx],
                'mel': self.mel_list[idx],
                'D': self.duration_list[idx],
                'f0': self.f0_list[idx],
                'std_f0': self.std_f0_list[idx]}

    def collate_fn(self, batch):
        text = [dic['text'] for dic in batch]
        mel = [dic['mel'] for dic in batch]
        duration = [dic['D'] for dic in batch]
        f0 = [dic['f0'] for dic in batch]
        std_f0 = [dic['std_f0'] for dic in batch]

        length_phone = np.array([p.shape[0] for p in text])
        max_src_len = np.max(length_phone).astype(int)

        length_mel = np.array([m.shape[0] for m in mel])
        max_mel_len = np.max(length_mel).astype(int)

        # sil_mat = silence_matrix(text, duration)

        text = pad_1d(text, max_src_len)
        duration = pad_1d(duration, np.max([d.shape[0] for d in duration]))
        mel = pad_2d(mel, max_mel_len)
        # sil_mat = pad_1d(sil_mat, max_mel_len)
        f0 = pad_1d(f0, np.max([f.shape[0] for f in f0]))
        std_f0 = pad_1d(std_f0, np.max([f.shape[0] for f in std_f0]))
        log_duration = np.log(duration + self.log_offset)

        phone_mask = length_mask(length_phone, max_src_len)
        mel_mask = length_mask(length_mel, max_mel_len)

        # sil_mask = text == 1
        # phone_mask = phone_mask | sil_mask
        #
        # sil_mask = sil_mat == 1
        # mel_mask = mel_mask | sil_mask

        f0 = np.round(f0)

        text = torch.from_numpy(text)
        mel = torch.from_numpy(mel).float()
        duration = torch.from_numpy(duration).long()
        log_duration = torch.from_numpy(log_duration).float()
        f0 = torch.from_numpy(f0).long()
        std_f0 = torch.from_numpy(std_f0).float()
        max_mel_len = torch.tensor(max_mel_len)
        phone_mask = torch.tensor(phone_mask)
        mel_mask = torch.tensor(mel_mask)

        return text, mel, duration, log_duration, f0, std_f0, phone_mask, mel_mask, max_mel_len
