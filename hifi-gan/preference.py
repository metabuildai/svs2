from dataclasses import dataclass, field
from typing import List
import json


@dataclass
class Config:
    # preprocess
    wave_file: List[str] = field(default_factory=list)

    # train dataset
    preprocessed_dir: str = ''
    train_file: List[str] = field(default_factory=list)
    valid_file: List[str] = field(default_factory=list)

    # audio parameter
    audio_segment_size: int = 8192
    num_mel: int = 80
    n_fft: int = 1024
    hop_size: int = 256
    win_size: int = 1024
    sampling_rate: int = 44100
    max_wave_value: float = 32768.0

    # dataset
    shuffle: bool = True
    batch_size: int = 16
    num_workers: int = 1
    pin_memory: bool = True
    drop_last: bool = True

    # train
    gpu_index: int = 0
    learning_rate: float = 0.0001
    adam_b1: float = 0.8
    adam_b2: float = 0.99
    lr_decay: float = 0.999
    training_epoch: int = 100
    last_epoch: int = 0
    checkpoint_dir: str = './checkpoint'
    checkpoint_interval: int = 3
    last_checkpoint_file: str = ''
    use_valid: bool = False
    use_log: bool = False
    log_dir: str = './log'
    train_log: str = ''
    valid_log: str = ''
    remove_checkpoint: bool = False

    # model
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