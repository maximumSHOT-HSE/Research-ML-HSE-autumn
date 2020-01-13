import argparse
import copy

from model import VoiceActivityDetector

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Shows model parameters'
    )
    parser.add_argument(
        'model_path',
        type=str,
        help='The path where model will be saved'
    )
    args = parser.parse_args()

    detector = VoiceActivityDetector()
    detector.load(args.model_path)

    print(detector)
