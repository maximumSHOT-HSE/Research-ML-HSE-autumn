import argparse

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from model import VoiceActivityDetector
from utils import load_labeled_audio, build_spectrogram


class PredictVadDataset(Dataset):

    DATA_MODES = ['train', 'val', 'test']

    def __init__(self, spectrogram, sample_pxl_width, pxl_ls):
        super().__init__()
        self.spectrogram = spectrogram
        self.sample_pxl_width = sample_pxl_width
        self.pxl_ls = pxl_ls
        self.len = len(pxl_ls)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        pxl_l = self.pxl_ls[index]
        x = spectrogram[:, pxl_l: pxl_l + sample_pxl_width, :]
        x = Image.fromarray(x)
        x = VoiceActivityDetector.from_picture_to_tensor(x)
        return x


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Voice Activity Detector'
    )
    parser.add_argument(
        '--graph-path',
        type=str,
        help='Path to the file where labeled graph will be stored'
    )
    parser.add_argument(
        'model_path',
        type=str,
        help='The path where model is stored'
    )
    parser.add_argument(
        'audio_path',
        type=str,
        help='The path to the audio file to be processed'
    )
    parser.add_argument(
        '--mat-output-path',
        type=str,
        help='The path to the .mat file where labels will be stored'
    )
    parser.add_argument(
        '--device',
        type=str,
        help='Device type for computations',
        default='cuda'
    )
    parser.add_argument(
        '--statistics-path',
        type=str,
        help='The path to the file where statistics of processing will be stored'
    )
    args = parser.parse_args()

    # ========================================================
    detector = VoiceActivityDetector()
    detector.load(args.model_path)

    rate, signal, labels = load_labeled_audio(args.audio_path)

    signal = signal[int(300 * rate): int(350 * rate)]
    labels = labels[int(300 * rate): int(350 * rate)]
    ts = np.linspace(0, len(signal) / rate, num=len(signal))

    # ========================================================
    spectrogram = build_spectrogram(
        signal,
        rate,
        n_filters=detector.params['n_filters'],
        window_size_s=detector.params['window_size'],
        step_size_ratio=detector.params['step_size_ratio']
    )
    # ========================================================
    net_window_size_f = int(rate * detector.params['net_window_size'])
    net_step_size_f = int(net_window_size_f * detector.params['net_step_size_ratio'])

    signal_size_f = len(signal)

    spectrogram_pxl_width = spectrogram.shape[1]
    frames_to_pixels_ratio = spectrogram_pxl_width / signal_size_f
    sample_pxl_width = detector.params['net_window_size'] * rate * frames_to_pixels_ratio
    sample_pxl_width = int(np.ceil(sample_pxl_width))
    # ========================================================
    pxl_ls = []
    ls = []
    for l in range(0, signal_size_f - net_window_size_f + 1, net_step_size_f):
        r = l + net_window_size_f
        # [l, r)

        pxl_l = int(np.ceil(l * frames_to_pixels_ratio))
        pxl_r = pxl_l + sample_pxl_width
        if pxl_r > spectrogram_pxl_width:
            break

        pxl_ls.append(pxl_l)
        ls.append(l)

    dataset = PredictVadDataset(spectrogram, sample_pxl_width, pxl_ls)
    data_loader = DataLoader(dataset, batch_size=40, shuffle=False)

    pred_labels = detector.predict(data_loader)

    total = np.zeros(len(signal) + 1)
    speech = np.zeros(len(signal) + 1)

    for l, pred_label in zip(ls, pred_labels):
        r = l + net_window_size_f
        total[l] += 1
        total[r] -= 1
        if pred_label:
            speech[l] += 1
            speech[r] -= 1

    speech = np.cumsum(speech)[:-1]
    total = np.cumsum(total)[:-1]
    prediction = (speech / total > 0.5).astype(int).reshape(-1, 1)

    print(f'test accuracy = {100 * np.sum(prediction == labels) / len(prediction)}%')

    # ========================================================
    plt.figure(figsize=(20, 20))

    plt.subplot(211)
    plt.plot(ts, signal, label='signal')
    plt.plot(ts, labels * 0.1 + 1.3, label='ground truth')
    plt.plot(ts, prediction * 0.1 + 1.1, label='prediction')
    plt.subplot(212)
    plt.imshow(spectrogram)
    plt.legend()

    plt.show()
