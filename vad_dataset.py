import os

import numpy as np
import torch.utils.data

WAV_EXTENSION = '.wav'


def is_audio_file(path: str):
    return os.path.isfile(path) and path.endswith(WAV_EXTENSION)


def seconds_to_frames(s, rate):
    return int(np.round(s * rate))


class VadDataset(torch.utils.data.Dataset):

    """
    Data will be loaded in the following way:
        the main directory with data is root
        audio files in the root with .wav extension and
        all of them has the file with the same name, but with .mat extension,
        which is file with labels of all frames in corresponding audio file
        audio files are sorted by the name, after that all audio files
        are divided into pieces of the same size (maybe except of the last piece)
        then each piece will be labeled with the majority of labels of all frames
        in piece (0=noise/1=noise in noise). Each piece has its own unique index.
        Pieces are numbered in the following way:
            0, 1, ..., n1 - 1
            n1, n1 + 1, ..., n1 + n2 - 1,
            ...= DataLo
        Indexation and information about directory content will be fixed
        after creating DataLoader object
    Args:
        root: path of directory with audio files and labels
        piece_size_s: float is the size of piece in seconds
    """

    def __init__(self, root, piece_size_s):
        self.root = root
        self.piece_size = piece_size_s
        self.audio_files = []
        # for file in sorted(os.listdir(root)):
        #     file_path = os.path.abspath(os.path.join(root, file))
        #     if is_audio_file(file_path):
        #         sample_rate, signal = scipy.io.wavfile.read(file_path)
        #         piece_count = int(np.ceil(len(signal) / seconds_to_frames(piece_size_s, sample_rate)))
        #         self.audio_files.append((file_path, piece_count))
        # print(self.audio_files)

    def __getitem__(self, item):
        pass
