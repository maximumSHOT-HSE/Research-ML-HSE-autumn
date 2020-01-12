import argparse

import numpy as np

from model import VoiceActivityDetector
from utils import build_spectrogram, load_labeled_audio, save_images

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=
        """
        Script for converting raw audio file into convenient labeled dataset.
        New files will be added into directory corresponding to the label of sample (speech/noise).
        """
    )
    parser.add_argument(
        'model_path',
        type=str,
        help='The path where model is stored'
    )
    parser.add_argument(
        '--max-speech-samples',
        type=int,
        default=1000,
        help='The maximum number of samples with speech label to be saved'
    )
    parser.add_argument(
        '--max-noise-samples',
        type=int,
        default=1500,
        help='The maximum number of samples with noise label to be saved'
    )
    parser.add_argument(
        'audio_path',
        type=str,
        help='Path to the raw audio file in .wav format'
    )
    parser.add_argument(
        'dataset_dir',
        type=str,
        help='Path to the directory, where samples will be stored'
    )
    parser.add_argument(
        '--shuffle',
        type=bool,
        help='The flag for samples shuffling before saving',
        default=True
    )
    args = parser.parse_args()

    detector = VoiceActivityDetector()
    detector.load(args.model_path)

    rate, signal, labels = load_labeled_audio(args.audio_path)

    X = int(rate * 60 * 10)
    signal = signal[: X]
    labels = labels[: X]

    pref_sum_labels = np.cumsum(labels)

    net_window_size_f = int(rate * detector.params['net_window_size'])
    net_step_size_f = int(net_window_size_f * detector.params['net_step_size_ratio'])

    signal_size_f = len(signal)

    speech_images = []
    noise_images = []

    spectrogram = build_spectrogram(
        signal,
        rate,
        n_filters=detector.params['n_filters'],
        window_size_s=detector.params['window_size'],
        step_size_ratio=detector.params['step_size_ratio']
    )

    spectrogram_pxl_width = spectrogram.shape[1]
    frames_to_pixels_ratio = spectrogram_pxl_width / signal_size_f
    sample_pxl_width = detector.params['net_window_size'] * rate * frames_to_pixels_ratio
    sample_pxl_width = int(np.ceil(sample_pxl_width))

    for l in range(0, signal_size_f - net_window_size_f + 1, net_step_size_f):
        r = l + net_window_size_f
        # [l, r)

        percentage = (pref_sum_labels[r - 1] - (pref_sum_labels[l - 1] if l > 0 else 0)) / (r - l)
        img_label = int(percentage > 0.5)

        pxl_l = int(np.ceil(l * frames_to_pixels_ratio))
        pxl_r = pxl_l + sample_pxl_width
        if pxl_r > spectrogram_pxl_width:
            break

        if img_label:  # speech
            speech_images.append(pxl_l)
        else:  # noise
            noise_images.append(pxl_l)

    if args.shuffle:
        print('Shuffling...')
        np.random.shuffle(speech_images)
        np.random.shuffle(noise_images)

    speech_images = speech_images[: min(len(speech_images), args.max_speech_samples)]
    noise_images = noise_images[: min(len(noise_images), args.max_noise_samples)]

    save_images(noise_images, speech_images, args.dataset_dir, spectrogram, sample_pxl_width)
