import argparse

import matplotlib.pyplot as plt
import numpy as np

# from model import VoiceActivityDetector
from utils import load_labeled_audio, build_spectrogram
# import torch


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

    # detector = VoiceActivityDetector()
    # detector.load(args.model_path)

    rate, signal, labels = load_labeled_audio(args.audio_path)

    signal = signal[0: int(100 * rate)]
    labels = labels[0: int(100 * rate)]
    ts = np.linspace(0, len(signal) / rate, num=len(signal))

    # spectrogram = build_spectrogram(
    #     signal,
    #     rate,
    #     n_filters=detector.params['n_filters'],
    #     window_size_s=detector.params['window_size'],
    #     step_size_ratio=detector.params['step_size_ratio']
    # )

    plt.plot(ts, signal, label='signal')
    plt.plot(ts, labels * 0.1 + 1, label='ground truth')

    plt.show()
