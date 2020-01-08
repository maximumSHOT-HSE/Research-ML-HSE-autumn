import argparse

import matplotlib.pyplot as plt
import numpy as np

from model import VoiceActivityDetector
from utils import load_labeled_audio

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Real Time Voice Activity Detector'
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
    parser.add_argument(
        '--buffer-size',
        type=float,
        default=0.05,
        help='The buffer size for audio pieces in seconds'
    )
    args = parser.parse_args()

    # ========================================================
    detector = VoiceActivityDetector()
    detector.load(args.model_path)

    rate, signal, labels = load_labeled_audio(args.audio_path)

    # signal = signal[int(300 * rate): int(350 * rate)]
    # labels = labels[int(300 * rate): int(350 * rate)]
    ts = np.linspace(0, len(signal) / rate, num=len(signal))

    detector.eval(rate)

    buffer_size_f = int(np.ceil(rate * args.buffer_size))
    signal_size_f = len(signal)

    pred = np.array([], dtype=int)

    for i in range(0, signal_size_f, buffer_size_f):
        piece = signal[i: min(i + buffer_size_f, signal_size_f)]
        detector.append(piece, i + buffer_size_f >= signal_size_f)
        pred = np.append(pred, detector.query())

    detector.flush_predictions()
    pred = np.append(pred, detector.query())

    labels = labels.reshape(-1)
    pred = pred.reshape(-1)

    print(np.sum(labels == pred) / len(labels))

    # plt.figure(figsize=(20, 20))
    #
    # plt.plot(ts, signal, label='signal')
    # plt.plot(ts, labels * 0.1 + 1.3, label='ground truth')
    # plt.plot(ts, pred * 0.1 + 1.1, label='prediction')
    #
    # plt.legend()
    #
    # plt.show()
