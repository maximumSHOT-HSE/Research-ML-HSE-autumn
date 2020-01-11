import numpy as np


class StreamBuffer:

    def __init__(self, rate):
        self.rate = rate

        self.frames_buffer = np.array([], dtype=np.float64)
        self.speech_votes = np.array([], dtype=np.float64)
        self.total_votes = np.array([], dtype=np.float64)
        self.labels = np.array([], dtype=int)

        self.spectrogram = None
        self.last_prev_frame_signal = 0

    def append(self, add_frames):
        self.frames_buffer = np.append(self.frames_buffer, add_frames)
        self.speech_votes = np.append(self.speech_votes, np.zeros_like(add_frames))
        self.total_votes = np.append(self.total_votes, np.zeros_like(add_frames))
