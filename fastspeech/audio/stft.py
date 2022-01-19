import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from scipy.signal import get_window
from librosa.util import pad_center
from librosa.filters import mel as librosa_mel_fn
from audio.audio_processing import dynamic_range_compression
from audio.audio_processing import dynamic_range_decompression
from librosa import filters


class STFT(torch.nn.Module):
    def __init__(self, filter_length, hop_length, win_length,
                 window='hann'):
        super(STFT, self).__init__()
        self.filter_length = filter_length
        self.hop_length = hop_length
        self.win_length = win_length
        self.window = window
        self.forward_transform = None
        scale = self.filter_length / self.hop_length
        fourier_basis = np.fft.fft(np.eye(self.filter_length))

        cutoff = int((self.filter_length / 2 + 1))
        fourier_basis = np.vstack([np.real(fourier_basis[:cutoff, :]),
                                   np.imag(fourier_basis[:cutoff, :])])

        forward_basis = torch.FloatTensor(fourier_basis[:, None, :])
        inverse_basis = torch.FloatTensor(
            np.linalg.pinv(scale * fourier_basis).T[:, None, :])

        if window is not None:
            assert(filter_length >= win_length)
            # get window and zero center pad it to filter_length
            fft_window = get_window(window, win_length, fftbins=True)
            fft_window = pad_center(fft_window, filter_length)
            fft_window = torch.from_numpy(fft_window).float()

            # window the bases
            forward_basis *= fft_window
            inverse_basis *= fft_window

        self.register_buffer('forward_basis', forward_basis.float())
        self.register_buffer('inverse_basis', inverse_basis.float())

    def transform(self, input_data):
        num_batches = input_data.size(0)
        num_samples = input_data.size(1)

        self.num_samples = num_samples

        # similar to librosa, reflect-pad the input
        input_data = input_data.view(num_batches, 1, num_samples)
        input_data = F.pad(
            input_data.unsqueeze(1),
            (int(self.filter_length / 2), int(self.filter_length / 2), 0, 0),
            mode='reflect')
        input_data = input_data.squeeze(1)

        forward_transform = F.conv1d(
            input_data.cuda(),
            Variable(self.forward_basis, requires_grad=False).cuda(),
            stride=self.hop_length,
            padding=0).cpu()

        cutoff = int((self.filter_length / 2) + 1)
        real_part = forward_transform[:, :cutoff, :]
        imag_part = forward_transform[:, cutoff:, :]

        magnitude = torch.sqrt(real_part**2 + imag_part**2)
        phase = torch.autograd.Variable(
            torch.atan2(imag_part.data, real_part.data))

        return magnitude, phase


class TacotronSTFT(torch.nn.Module):
    def __init__(self, filter_length, hop_length, win_length,
                 n_mel_channels, sampling_rate, mel_fmin=0.0,
                 mel_fmax=8000.0):
        super(TacotronSTFT, self).__init__()
        self.n_mel_channels = n_mel_channels
        self.sampling_rate = sampling_rate
        self.stft_fn = STFT(filter_length, hop_length, win_length)
        mel_basis = librosa_mel_fn(
            sampling_rate, filter_length, n_mel_channels, mel_fmin, mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer('mel_basis', mel_basis)

    def spectral_normalize(self, magnitudes):
        output = dynamic_range_compression(magnitudes)
        return output

    def spectral_de_normalize(self, magnitudes):
        output = dynamic_range_decompression(magnitudes)
        return output

    def mel_spectrogram(self, y):
        assert(torch.min(y.data) >= -1)
        assert(torch.max(y.data) <= 1)

        magnitudes, phases = self.stft_fn.transform(y)
        magnitudes = magnitudes.data
        mel_output = torch.matmul(self.mel_basis, magnitudes)
        mel_output = self.spectral_normalize(mel_output)
        energy = torch.norm(magnitudes, dim=1)
         
        return mel_output, energy


def mel_spectrogram(y, mel_basis, hann_window, n_fft, hop_size, win_size, center=False):
    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window,
                      center=center, pad_mode='reflect', normalized=False, onesided=True)

    spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    spec = torch.matmul(mel_basis, spec)
    spec = torch.log(torch.clamp(spec, min=1e-5) * 1)
    energy = torch.norm(spec, dim=1)

    return spec, energy


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
        x = F.pad(x.unsqueeze(1), (int((self.fft - self.hop_size) / 2), int((self.fft - self.hop_size) / 2)),
                  mode='reflect')
        spec = torch.stft(x.squeeze(1), self.fft, hop_length=self.hop_size, win_length=self.win_size,
                          window=self.window, center=False, onesided=True)
        spec = torch.sqrt(spec.pow(2).sum(-1) + (1e-9))
        spec = torch.matmul(self.mel_basis, spec)

        return torch.log(torch.clamp(spec, min=1e-5))
