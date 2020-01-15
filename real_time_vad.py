import argparse
import time

import numpy as np
import torch
from tqdm import tqdm

from StreamBuffer import StreamBuffer
from model import VoiceActivityDetector
from utils import load_labeled_audio


def remove_short_pauses(signal, l):
    result = np.copy(signal)
    n = len(result)
    same = 0
    previous = signal[0]
    for i in range(n):
        if i > 0 and signal[i] == signal[i - 1]:
            same += 1
        else:
            same = 1

        if same < l:
            result[i] = previous
        else:
            previous = result[i]

    return result


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
        '--cuda',
        type=bool,
        default=False
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

    print(args.cuda)

    if args.cuda and torch.cuda.is_available():
        print('???')
        VoiceActivityDetector.DEVICE = torch.device('cuda')
    else:
        VoiceActivityDetector.DEVICE = torch.device('cpu')

    print(f'Processing on: {VoiceActivityDetector.DEVICE}')

    # ========================================================
    detector = VoiceActivityDetector()
    detector.load(args.model_path)

    rate, signal, labels = load_labeled_audio(args.audio_path)

    # signal = signal[int(500 * rate): int(750 * rate)]
    # labels = labels[int(500 * rate): int(750 * rate)]

    ts = np.linspace(0, len(signal) / rate, num=len(signal))

    detector.setup(rate)
    stream_buffer = StreamBuffer(rate)

    buffer_size_f = int(np.ceil(rate * args.buffer_size))
    signal_size_f = len(signal)

    pred = np.array([], dtype=int)

    history = []

    for i in tqdm(range(0, signal_size_f, buffer_size_f)):
        piece = signal[i: min(i + buffer_size_f, signal_size_f)]

        st = time.time()
        detector.append(piece, stream_buffer)
        pred = np.append(pred, detector.query(stream_buffer))
        fn = time.time()

        history.append(fn - st)

    detector.flush_predictions(stream_buffer)
    pred = np.append(pred, detector.query(stream_buffer))

    labels = labels.reshape(-1)
    pred = pred.reshape(-1)

    pred = remove_short_pauses(pred, int(rate * 0.2))

    tp = 1
    tn = 1
    fp = 1
    fn = 1

    for y_target, y_pred in zip(labels, pred):
        if y_target == 1:
            if y_pred == 1:
                tp += 1
            else:
                fn += 1
        else:
            if y_pred == 1:
                fp += 1
            else:
                tn += 1

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    fpr = fp / (fp + tn)
    accuracy = (tp + tn) / (tp + fp + tn + fn)

    print(f'precision = {precision}')
    print(f'recall = {recall}')
    print(f'fpr = {fpr}')
    print(f'accuracy = {accuracy}')
