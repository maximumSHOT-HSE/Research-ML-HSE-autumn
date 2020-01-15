import argparse
import typing
from pathlib import Path

import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader

import utils
from model import VoiceActivityDetector


class TrainVadDataset(Dataset):

    DATA_MODES = ['train', 'val', 'test']

    def __init__(self, files: typing.List[Path], mode):
        super().__init__()
        self.files = sorted(files)

        if mode not in TrainVadDataset.DATA_MODES:
            raise Exception(f'Unrecognized mode = {mode}')

        self.mode = mode
        self.len = len(files)

        if self.mode != 'test':
            self.labels = [path.parent.name for path in self.files]

    def __len__(self):
        return self.len

    def load_sample(self, path):
        image = Image.open(path)
        image.load()
        return image

    def __getitem__(self, index):
        x = self.load_sample(self.files[index])
        x = VoiceActivityDetector.from_picture_to_tensor(x)
        if self.mode == 'test':
            return x
        else:
            label_id = VoiceActivityDetector.LABEL_TO_IDX[self.labels[index]]
            return x, label_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        """
        Script for model training
        """
    )
    parser.add_argument(
        '--verbose',
        type=bool,
        default=True,
        help='Toggle for printing info about training'
    )
    parser.add_argument(
        'model_path',
        type=str,
        help='The path where model is stored'
    )
    parser.add_argument(
        'sessions_dir',
        type=str,
        help='The path to the directory, where new session will be stored.\n'
             'session_<session_id>/checkpoint_<session_id>_<checkpoint_id>.vad'
    )
    parser.add_argument(
        'dataset_dir',
        type=str,
        help='The path of directory with training dataset'
    )
    parser.add_argument(
        '--statistics',
        type=bool,
        default=False
    )
    parser.add_argument(
        '--graph',
        type=bool,
        default=False
    )
    parser.add_argument(
        '--val-ratio',
        type=float,
        default=0.2,
        help='The ratio of dataset for validation part'
    )
    parser.add_argument(
        '--test-ratio',
        type=float,
        default=0.1,
        help='The ratio of dataset for testing part'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=40,
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=10
    )
    parser.add_argument(
        '--cuda',
        type=bool,
        default=True
    )
    args = parser.parse_args()

    if args.cuda and torch.cuda.is_available():
        VoiceActivityDetector.DEVICE = torch.device('cuda')
    else:
        VoiceActivityDetector.DEVICE = torch.device('cpu')

    print(f'Processing on: {VoiceActivityDetector.DEVICE}')

    detector = VoiceActivityDetector()
    detector.load(args.model_path)

    dataset_dir = Path(args.dataset_dir)
    dataset_paths = sorted(list(dataset_dir.rglob('*.png')))
    labels = [path.parent.name for path in dataset_paths]

    X_train, X_val, y_train, y_val = train_test_split(
        dataset_paths,
        labels,
        test_size=args.val_ratio,
        shuffle=True
    )

    train_dataset = TrainVadDataset(X_train, mode='train')
    val_dataset = TrainVadDataset(X_val, mode='val')

    train_loader = DataLoader(train_dataset, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, args.batch_size, shuffle=False)

    session_id = utils.get_max_session_id(args.sessions_dir) + 1

    history = detector.fit(train_loader, val_loader, args.epochs, args.sessions_dir, session_id, args.verbose)

    if args.graph:
        utils.save_train_graph(args.sessions_dir, session_id, history)

    if args.statistics:
        utils.save_statistics(args.sessions_dir, session_id, history)
