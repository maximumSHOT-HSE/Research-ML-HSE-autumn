import argparse
import copy

from model import VoiceActivityDetector

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Creates a voice activity detector with given parameters'
    )
    parser.add_argument(
        '--window-size',
        type=float,
        default=0.025,
        help='Windows size in seconds'
    )
    parser.add_argument(
        '--step-size-ratio',
        type=float,
        default=0.5,
        help='Step size ratio: percentage of window size in [0, 1]'
    )
    parser.add_argument(
        '--n-filters',
        type=int,
        default=60,
        help='The number of filters in spectrogram'
    )
    parser.add_argument(
        '--net-window-size',
        type=float,
        default=0.05,
        help='Window size of neural network in seconds'
    )
    parser.add_argument(
        '--net-step-size-ratio',
        type=float,
        default=0.5,
        help='Step size ratio of neural network: percentage of window size for neural network in [0, 1]'
    )
    parser.add_argument(
        'model_path',
        type=str,
        help='The path where model will be saved'
    )
    parser.add_argument(
        '--arc',
        type=str,
        default='cnn',
        help='Architecture type of neural network'
    )
    args = parser.parse_args()

    params: dict = copy.deepcopy(vars(args))

    params.pop('model_path')
    params.pop('arc')  # TODO: process different architectures

    detector = VoiceActivityDetector(params)

    print(f'Saving model...\n{detector}')
    detector.save(args.model_path)
    print('Done')
