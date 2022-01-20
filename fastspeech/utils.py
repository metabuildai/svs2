import torch
import torch.nn.functional as F
import numpy as np
import soundfile


def get_alignment(tier):
    sampling_rate = 22050
    hop_length = 256
    sil_phones = ['sil', 'sp', 'spn']

    phones = []
    durations = []
    start_time = 0
    end_time = 0
    end_idx = 0
    for t in tier._objects:
        s, e, p = t.start_time, t.end_time, t.text

        # Trimming leading silences
        if phones == []:
            if p in sil_phones:
                continue
            else:
                start_time = s
        if p not in sil_phones:
            phones.append(p)
            end_time = e
            end_idx = len(phones)
        else:
            phones.append(p)
        durations.append(int(e*sampling_rate/hop_length)-int(s*sampling_rate/hop_length))

    # Trimming tailing silences
    phones = phones[:end_idx]
    durations = durations[:end_idx]
    
    return phones, np.array(durations), start_time, end_time


def get_mask_from_lengths(lengths, max_len=None):
    device = lengths.device

    batch_size = lengths.shape[0]
    if max_len is None:
        max_len = int(torch.max(lengths).item())

    ids = torch.arange(0, max_len).unsqueeze(0).expand(batch_size, -1).to(device)
    mask = (ids >= lengths.unsqueeze(1).expand(-1, max_len))

    return mask


def vocgan_infer(mel, vocoder, path):
    model = vocoder

    with torch.no_grad():
        if len(mel.shape) == 2:
            mel = mel.unsqueeze(0)

        audio = model.infer(mel).squeeze()

        audio = audio.cpu().numpy()
        soundfile.write(path, audio, 22050)
        # audio = hp.max_wav_value * audio[:-(hp.hop_length*10)]
        # audio = audio.clamp(min=-hp.max_wav_value, max=hp.max_wav_value-1)
        # audio = audio.short().cpu().detach().numpy()
        # wavfile.write(path, hp.sampling_rate, audio)


def pad(input_ele, mel_max_length=None):
    if mel_max_length:
        max_len = mel_max_length
    else:
        max_len = max([input_ele[i].size(0) for i in range(len(input_ele))])

    out_list = list()
    for i, batch in enumerate(input_ele):
        if len(batch.shape) == 1:
            one_batch_padded = F.pad(
                batch, (0, max_len-batch.size(0)), "constant", 0.0)
        elif len(batch.shape) == 2:
            one_batch_padded = F.pad(
                batch, (0, 0, 0, max_len-batch.size(0)), "constant", 0.0)
        out_list.append(one_batch_padded)
    out_padded = torch.stack(out_list)
    return out_padded


def _is_outlier(x, p25, p75):
    """Check if value is an outlier."""
    lower = p25 - 1.5 * (p75 - p25)
    upper = p75 + 1.5 * (p75 - p25)

    return np.logical_or(x <= lower, x >= upper)


def remove_outlier(x):
    """Remove outlier from x."""
    p25 = np.percentile(x, 25)
    p75 = np.percentile(x, 75)

    indices_of_outliers = []
    for ind, value in enumerate(x):
        if _is_outlier(value, p25, p75):
            indices_of_outliers.append(ind)

    x[indices_of_outliers] = 0.0

    # replace by mean f0.
    x[indices_of_outliers] = np.max(x)
    return x


def average_by_duration(x, durs):
    mel_len = durs.sum()
    durs_cum = np.cumsum(np.pad(durs, (1, 0)))

    # calculate charactor f0/energy
    x_char = np.zeros((durs.shape[0],), dtype=np.float32)
    for idx, start, end in zip(range(mel_len), durs_cum[:-1], durs_cum[1:]):
        values = x[start:end][np.where(x[start:end] != 0.0)[0]]
        x_char[idx] = np.mean(values) if len(values) > 0 else 0.0  # np.mean([]) = nan.

    return x_char.astype(np.float32)
