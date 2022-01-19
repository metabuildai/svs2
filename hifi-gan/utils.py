import torch
from torch.nn import functional as F
from librosa import filters


class MelSpectrogram(torch.nn.Module):
    def __init__(self, sr, fft, mel_channel, win_size, hop_size, fmin=0.0, fmax=None):
        super(MelSpectrogram, self).__init__()
        self.hop_size = hop_size
        self.fft = fft
        self.win_size = win_size
        mel_basis = filters.mel(sr, fft, mel_channel, fmin, fmax)
        self.mel_basis = torch.nn.Parameter(torch.from_numpy(mel_basis).float(), requires_grad=False)
        self.window = torch.nn.Parameter(torch.hann_window(win_size), requires_grad=False)

    def forward(self, x):
        x = F.pad(x.unsqueeze(1), (int((self.fft-self.hop_size)/2), int((self.fft-self.hop_size)/2)), mode='reflect')
        spec = torch.stft(x.squeeze(1), self.fft, hop_length=self.hop_size, win_length=self.win_size, window=self.window, center=False, onesided=True)
        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
        spec = torch.matmul(self.mel_basis, spec)

        return torch.log(torch.clamp(spec, min=1e-5))
