from dataclasses import dataclass, field
from typing import List, Tuple
import json


@dataclass
class Config:
    # preprocess
    preprocess_dir: str = './preprocess'
    wave_file: List[str] = field(default_factory=list)
    textgrid_file: List[str] = field(default_factory=list)

    max_duration: int = 0
    max_pitch: int = 0

    # train dataset
    train_file: List[str] = field(default_factory=list)
    valid_file: List[str] = field(default_factory=list)
    mel_scaler: str = ''
    f0_scaler: str = ''
    energy_scaler: str = ''

    # dataset
    num_workers: int = 4
    shuffle: bool = True
    pin_memory: bool = True
    drop_last: bool = True
    persistent_workers: bool = True
    log_offset: float = 1.0

    # train
    gpu_index: int = 0
    batch_size: int = 16
    learning_rate: float = 0.0001
    training_epoch: int = 1000
    last_epoch: int = 0
    checkpoint_dir: str = './checkpoint'
    checkpoint_interval: int = 3600
    last_checkpoint_file: str = ''
    model_path: str = ''
    log: str = './log'
    train_log: str = ''
    valid_log: str = ''

    # model
    encoder_head: int = 2
    fft_filter_size: int = 1024
    fft_kernel_size = (9, 1)
    encoder_dropout: float = 0.2
    decoder_head: int = 2
    decoder_dropout: float = 0.2
    variance_predictor_filter_size: int = 256
    variance_predictor_kernel_size: int = 3
    variance_predictor_dropout: float = 0.5

    # loss
    encoder_hidden: int = 256
    decoder_hidden: int = 256
    n_warm_up_step: int = 4000
    step: int = 0

    # optimizer
    betas: Tuple = (0.9, 0.98)
    eps: float = 1e-9
    weight_decay: float = 0.
    grad_clip_thresh: float = 1.0

    # audio
    sampling_rate = 44100
    filter_length = 1024
    hop_length = 256
    win_length = 1024
    max_wav_value = 32768.0
    n_mel_channels = 80
    mel_fmin = 0
    mel_fmax = None

    max_seq_len: int = 1100

    # hifi-gan
    upsample_rate = [8, 8, 2, 2]
    upsample_kernel_size = [16, 16, 4, 4]
    upsample_initial_channel: int = 512
    resblock_kernel_size = [3, 7, 11]
    resblock_dilation_size = [[1, 3, 5], [1, 3, 5], [1, 3, 5]]


def save(cf: Config, path: str):
    with open(path, 'w') as f:
        json.dump(cf.__dict__, f, indent=2)


def load(path: str):
    with open(path, 'r') as f:
        json_obj = json.load(f)
        config = Config(**json_obj)
    return config
