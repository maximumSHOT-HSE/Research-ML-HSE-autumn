from PIL import Image
import scipy.io.wavfile
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import utils
import PIL.Image

from model import VoiceActivityDetector

if __name__ == '__main__':

    LTIME = 0.5
    RTIME = LTIME + 1
    WINDOW_SIZE_S = 0.05
    STEP_SIZE_RATIO = 0.5

    rate, signal, labels = utils.load_labeled_audio('raw_audio_data/park.wav')

    signal = signal[int(LTIME * rate): int(RTIME * rate)]
    labels = labels[int(LTIME * rate): int(RTIME * rate)]

    ts = np.linspace(LTIME, RTIME, num=len(signal))

    plt.figure(figsize=(4, 3))
    plt.plot(ts, signal, label='сигнал')
    plt.plot(ts, labels * 0.1 + 0.75, label='речь')

    # plt.show()

    # plt.xlabel('время, сек')
    # plt.ylabel('магнитуда')

    # plt.legend()
    plt.show()

    # spectrogram = utils.build_spectrogram(
    #     signal,
    #     rate,
    #     n_filters=80,
    #     window_size_s=0.05,
    #     step_size_ratio=0.25
    # )
    #
    # plt.figure(figsize=(4, 3))
    #
    # plt.xticks([])
    # plt.yticks([])
    #
    # plt.imshow(spectrogram)
    # plt.show()
