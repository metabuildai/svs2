import torch
import torch.utils.data
from dataclasses import dataclass


@dataclass
class Segment:
    index: int
    start: int
    end: int


class Dataset(torch.utils.data.Dataset):
    def __init__(self, audio_list, audio_segment_size, hop_size):
        self.audio_list = audio_list
        self.segment_list = list()
        for i, audio in enumerate(audio_list):
            for start in range(0, audio.shape[0] - audio_segment_size, hop_size):
                self.segment_list.append(Segment(i, start, start + audio_segment_size))

    def __getitem__(self, index):
        segment = self.segment_list[index]
        return torch.from_numpy(self.audio_list[segment.index][segment.start:segment.end]).float()

    def __len__(self):
        return len(self.segment_list)
