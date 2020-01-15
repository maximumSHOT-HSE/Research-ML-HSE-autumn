import argparse
import time

import numpy as np
from tqdm import tqdm

from StreamBuffer import StreamBuffer
from model import VoiceActivityDetector
from utils import load_labeled_audio
import matplotlib.pyplot as plt


def predict(detector, signal, rate, buffer_size_s, artificial_sleep=False):
    stream_buffer = StreamBuffer(rate)
    buffer_size_f = int(np.ceil(rate * buffer_size_s))
    signal_size_f = len(signal)

    pred = np.array([], dtype=int)
    for i in tqdm(range(0, signal_size_f, buffer_size_f)):
        piece = signal[i: min(i + buffer_size_f, signal_size_f)]

        if artificial_sleep:
            time.sleep(buffer_size_s)

        detector.append(piece, stream_buffer)
        pred = np.append(pred, detector.query(stream_buffer))

    detector.flush_predictions(stream_buffer)
    pred = np.append(pred, detector.query(stream_buffer))

    return pred


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
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
    args = parser.parse_args()

    print(f'Processing on: {VoiceActivityDetector.DEVICE}')

    # ========================================================
    detector = VoiceActivityDetector()
    detector.load(args.model_path)

    rate, signal, labels = load_labeled_audio(args.audio_path)

    signal = signal[int(0 * rate): int(30 * rate)]
    labels = labels[int(0 * rate): int(30 * rate)]

    detector.setup(rate)

    buffer_sizes = list(range(20, 151, 5))  # ms
    ratios = []

    for buffer_size in buffer_sizes:
        print(f'buffer_size = {buffer_size}')
        st = time.time()
        pred = predict(detector, signal, rate, buffer_size / 1000, False)
        fn = time.time()
        ratios.append((len(signal) / rate) / (fn - st))

    plt.stem(buffer_sizes, ratios)
    plt.xlabel('size(packet), ms')
    plt.ylabel('size(signal) / (proc time)')
    plt.show()

    print(buffer_sizes)

    print(f'{fn - st}')
