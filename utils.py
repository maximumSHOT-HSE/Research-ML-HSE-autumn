import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io.wavfile
from PIL import Image
from matplotlib import cm


def hz_to_mel(C1, C2, f):
    return C1 * np.log10(1 + f / C2)


def mel_to_hz(C1, C2, m):
    return C2 * (10 ** (m / C1) - 1)


def find_nfft(window_size_f):
    nfft = 1
    while nfft < window_size_f:
        nfft *= 2
    return nfft


# takes signal, apply algorithm, returns image of shape WxHx3 with values in [0, 255]
def build_spectrogram(
        signal,
        rate,
        n_filters=40,
        window_size_s=0.025,
        step_size_ratio=0.5
):
    signal = np.append(signal[0], signal[1:] - 0.96 * signal[:-1])

    step_size_s = window_size_s * step_size_ratio

    signal_size_f = len(signal)
    window_size_f = int(window_size_s * rate)
    step_size_f = int(step_size_s * rate)
    num_windows = int(np.ceil(max(0, signal_size_f + 1 - window_size_f) / step_size_f)) + 1

    pad_signal_size_f = (num_windows - 1) * step_size_f + window_size_f
    pad_signal = np.append(signal, np.zeros((pad_signal_size_f - signal_size_f)))

    indices = np.tile(np.arange(window_size_f), (num_windows, 1)) + np.tile(
        np.arange(0, num_windows * step_size_f, step_size_f), (window_size_f, 1)).T
    windows = pad_signal[indices]
    windows *= np.hamming(window_size_f)

    n_fft = find_nfft(window_size_f)

    windows_mag = np.absolute(np.fft.rfft(windows, n_fft))  # magnitudes of FFT
    windows_power = (1.0 / n_fft) * (windows_mag ** 2)  # power spectrum

    C1 = 2595
    C2 = 700

    low_freq_mel = hz_to_mel(C1, C2, 0)
    high_freq_mel = hz_to_mel(C1, C2, rate / 2)

    mel_points = np.linspace(low_freq_mel, high_freq_mel, n_filters + 2)
    hz_points = mel_to_hz(C1, C2, mel_points)
    fs = (n_fft + 1) * hz_points / rate

    H = np.zeros((n_filters, int(np.floor(n_fft / 2)) + 1))
    for m in range(1, n_filters + 1):
        for k in range(H.shape[1]):
            if fs[m - 1] <= k < fs[m]:
                H[m - 1, k] = (k - fs[m - 1]) / (fs[m] - fs[m - 1])
            elif fs[m] < k <= fs[m + 1]:
                H[m - 1, k] = (fs[m + 1] - k) / (fs[m + 1] - fs[m])

    filter_banks = windows_power @ H.T
    filter_banks[filter_banks == 0] = np.finfo(float).eps
    filter_banks = 20 * np.log10(filter_banks)  # Hz

    # normalization
    filter_banks -= np.min(filter_banks)
    filter_banks /= np.max(filter_banks)
    # after: [0, 1]

    img = (cm.jet(filter_banks.T) * 255).astype(dtype='uint8')

    return np.clip(img[:, :, :3], 0, 255)


def get_max_session_id(sessions_dir):
    max_id = 0
    for path in os.listdir(sessions_dir):
        if os.path.isdir(os.path.join(sessions_dir, path)) and path.startswith('session'):
            max_id = max(max_id, int(path[len('session_'):]))
    return max_id


def get_max_checkpoint_id(session_dir):
    max_id = 0
    for path in os.listdir(session_dir):
        if path.startswith('checkpoint_'):
            id = int(path[path.rfind('_') + 1: -4])
            max_id = max(max_id, id)
    return max_id


def get_session_path(sessions_dir, session_id):
    return os.path.join(sessions_dir, f'session_{session_id}')


def save_model(sessions_dir, session_id, model):
    path = get_session_path(sessions_dir, session_id)
    if not os.path.isdir(path):
        os.mkdir(path)
    checkpoint_id = get_max_checkpoint_id(path) + 1
    filename = os.path.join(path, f'checkpoint_{session_id}_{checkpoint_id}.vad')
    model.save(filename)


def save_train_graph(session_dir, session_id, history):
    path = get_session_path(session_dir, session_id)
    path = os.path.join(path, 'graph')

    train_loss, train_accuracy, val_loss, val_accuracy = zip(*history)

    plt.figure(figsize=(15, 10))

    plt.subplot(211)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='val')

    plt.subplot(212)
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.plot(train_accuracy, label='train')
    plt.plot(val_accuracy, label='val')

    plt.legend()

    plt.savefig(path)


def save_statistics(session_dir, session_id, history):
    pass


def load_audio(path: str):
    rate, signal = scipy.io.wavfile.read(path)
    signal = np.copy(signal).astype(dtype=np.float32)

    _min = np.min(signal)
    signal -= _min
    _max = np.max(signal)
    if _max > 0:
        signal /= _max

    return rate, signal


# loads signal, normalize to [0, 1]
# returns (rate, signal, labels)
def load_labeled_audio(path: str):
    if not path.endswith('.wav'):
        raise Exception(f'Unrecognized audio format: expected .wav, but found = {path}')

    rate, signal = load_audio(path)
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
