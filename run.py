import scipy.io.wavfile
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import utils
import PIL.Image

from model import VoiceActivityDetector

if __name__ == '__main__':

    LTIME = 40.4
    RTIME = LTIME + 10

    rate, signal = scipy.io.wavfile.read('raw_audio_data/park.wav')
    signal = signal[int(LTIME * rate): int(RTIME * rate)]

    spectrogram = utils.build_spectrogram(signal, rate, n_filters=40)

    # img = (cm.jet(spectrogram.T) * 255).astype(dtype='uint8')


    tmp = VoiceActivityDetector.from_picture_to_tensor(spectrogram)

    print(spectrogram.shape)
    print()
    print(tmp.size())

    tmp = VoiceActivityDetector.from_tensor_to_picture(tmp)

    plt.subplot(211)
    plt.imshow(spectrogram)
    plt.subplot(212)
    plt.imshow(tmp)
    plt.show()

    # print(spectrogram)
    # print(spectrogram.min(), spectrogram.max())
    #
    # PIL.Image.fromarray(spectrogram).save('tmp.png')
