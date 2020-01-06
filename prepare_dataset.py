import argparse
import os

import numpy as np
import scipy.io
import scipy.io.wavfile
from PIL import Image

from model import VoiceActivityDetector
from utils import build_spectrogram


# loads signal, normalize to [0, 1]
# returns (rate, signal, labels)
def load_labeled_audio(path: str):
    if not path.endswith('.wav'):
        raise Exception(f'Unrecognized audio format: expected .wav, but found = {path}')

    rate, signal = scipy.io.wavfile.read(path)
    signal = np.copy(signal).astype(dtype=np.float32)

    _min = np.min(signal)
    signal -= _min
    _max = np.max(signal)
    if _max > 0:
        signal /= _max

    labels = scipy.io.loadmat(path[:-4] + '.mat')['y_label'].astype(np.long)

    if len(signal) != len(labels):
        raise Exception(f'Signal and labels should have equal lengths, but found '
                        f'signal length = {len(signal)}, labels length = {len(labels)}')

    return rate, signal, labels


def get_max_id_in_dir(dir_path):
    max_id = 0
    for file in os.listdir(dir_path):
        if os.path.isfile(os.path.join(dir_path, file)) and file.endswith('.png'):
            max_id = max(max_id, int(file[:-4]))
    return max_id


def save_images(noise_images, speech_images, dir_path, spectrogram, sample_pxl_width):
    noise_dir = os.path.join(dir_path, VoiceActivityDetector.IDX_TO_LABEL[0])
    speech_dir = os.path.join(dir_path, VoiceActivityDetector.IDX_TO_LABEL[1])

    print(f'Saving into\n\'{noise_dir}\' for noise\n\'{speech_dir}\' for speech\n'
          f'format: <id>.png\n')

    print(f'Image size (HxW) = {spectrogram.shape[0]}x{sample_pxl_width}')

    if not os.path.isdir(noise_dir):
        os.mkdir(noise_dir)
    if not os.path.isdir(speech_dir):
        os.mkdir(speech_dir)

    index = max(
        get_max_id_in_dir(noise_dir),
        get_max_id_in_dir(speech_dir)
    )

    for pxl_l in noise_images:
        index += 1
        path = os.path.join(noise_dir, str(index) + '.png')
        Image.fromarray(spectrogram[:, pxl_l: pxl_l + sample_pxl_width, :]).save(path)

    for pxl_l in speech_images:
        index += 1
        path = os.path.join(speech_dir, str(index) + '.png')
        Image.fromarray(spectrogram[:, pxl_l: pxl_l + sample_pxl_width, :]).save(path)


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
        'dir',
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

    for l in range(int(16000 * 3.6), signal_size_f - net_window_size_f + 1, net_step_size_f):
        r = l + net_window_size_f
        # [l, r)

        percentage = (pref_sum_labels[r - 1] - (pref_sum_labels[l - 1] if l > 0 else 0)) / (r - l)
        img_label = int(percentage > 0.5)

        pxl_l = int(np.ceil(l * frames_to_pixels_ratio))
        pxl_r = pxl_l + sample_pxl_width
        if pxl_r >= spectrogram_pxl_width:
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

    save_images(noise_images, speech_images, args.dir, spectrogram, sample_pxl_width)
