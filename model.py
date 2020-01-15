import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import utils
from cnn import CNN

import matplotlib.pyplot as plt
from real_time_vad_dataset import RealTimeVadDataset
from torch.utils.data import Dataset, DataLoader
from StreamBuffer import StreamBuffer


class VoiceActivityDetector:

    DEFAULT_RATE = 16000

    def __init__(self, params=None):
        self.params = params
        self.net = CNN()
        self.rate = VoiceActivityDetector.DEFAULT_RATE

    IDX_TO_LABEL = {
        0: 'noise',
        1: 'speech'
    }

    LABEL_TO_IDX = {
        'noise': 0,
        'speech': 1
    }

    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])
    DEVICE = torch.device('cpu')

    @staticmethod
    def from_picture_to_tensor(picture):
        picture = np.array(picture)[:, :, :3]
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(VoiceActivityDetector.MEAN, VoiceActivityDetector.STD)
        ])
        picture = Image.fromarray(picture).resize(CNN.IMG_SIZE)
        return transform(np.array(picture, dtype='float32') / 255)

    @staticmethod
    def from_tensor_to_picture(tensor):
        tensor = tensor.numpy().transpose((1, 2, 0))
        tensor = VoiceActivityDetector.STD * tensor + VoiceActivityDetector.MEAN
        tensor *= 255
        tensor = np.clip(tensor, 0, 255).astype(dtype=int)
        return tensor

    def save(self, path: str):
        torch.save(
            {
                'params': self.params,
                'model_state_dict': self.net.state_dict()
            },
            path
        )

    def load(self, path: str):
        dump = torch.load(path, map_location=VoiceActivityDetector.DEVICE)
        self.params = dump['params']
        self.net.load_state_dict(dump['model_state_dict'])

    def __str__(self):
        return str(self.params) + '\n' + str(self.net)

    def train_epoch(self, train_loader, optimizer, loss_f):
        self.net.train()

        cum_loss = 0.0
        samples_count = 0
        accuracy = 0

        for inputs, targets in train_loader:
            inputs = inputs.to(device=VoiceActivityDetector.DEVICE, dtype=torch.float)
            targets = targets.to(device=VoiceActivityDetector.DEVICE, dtype=torch.long)

            optimizer.zero_grad()
            outputs = self.net(inputs)
            loss = loss_f(outputs, targets)
            loss.backward()
            optimizer.step()
            train_preds = torch.argmax(outputs, 1)

            cum_loss += loss.item() * inputs.size(0)

            accuracy += torch.sum(train_preds == targets.data).item()
            samples_count += inputs.size(0)

        cum_loss /= samples_count
        accuracy /= samples_count

        return cum_loss, accuracy

    def validate_epoch(self, val_loader, loss_f):
        self.net.eval()

        cum_loss = 0.0
        samples_count = 0
        accuracy = 0

        for inputs, targets in val_loader:
            inputs = inputs.to(device=VoiceActivityDetector.DEVICE, dtype=torch.float)
            targets = targets.to(device=VoiceActivityDetector.DEVICE, dtype=torch.long)

            with torch.no_grad():
                outputs = self.net(inputs)
                loss = loss_f(outputs, targets)
                val_preds = torch.argmax(outputs, 1)

            cum_loss += loss.item() * inputs.size(0)
            accuracy += torch.sum(val_preds == targets.data).item()
            samples_count += inputs.size(0)

        cum_loss /= samples_count
        accuracy /= samples_count

        return cum_loss, accuracy

    def fit(self, train_loader, val_loader, epochs, sessions_dir, session_id, verbose=True):
        self.net = self.net.to(VoiceActivityDetector.DEVICE)

        optimizer = torch.optim.Adam(self.net.parameters())
        loss_f = nn.CrossEntropyLoss()

        history = []
        best_val_loss = np.inf

        for epoch in tqdm(range(epochs)):
            train_loss, train_accuracy = self.train_epoch(train_loader, optimizer, loss_f)
            val_loss, val_accuracy = self.validate_epoch(val_loader, loss_f)
            history.append((train_loss, train_accuracy, val_loss, val_accuracy))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                utils.save_model(sessions_dir, session_id, self)

            if verbose:
                print(f'epoch {epoch + 1}: train loss = {train_loss}, train accuracy = {train_accuracy}, '
                      f'val loss = {val_loss}, val accuracy = {val_accuracy}')

        return history

    def predict(self, data_loader):
        self.net.eval()
        self.net = self.net.to(VoiceActivityDetector.DEVICE)
        pred_labels = np.array([], dtype=int)

        for inputs in data_loader:
            inputs = inputs.to(device=VoiceActivityDetector.DEVICE, dtype=torch.float)

            with torch.no_grad():
                outputs = self.net(inputs)
                val_preds = torch.argmax(outputs, 1)
                val_preds = val_preds.cpu().numpy()
                pred_labels = np.append(pred_labels, val_preds)

        return pred_labels

    def setup(self, rate):
        self.rate = rate
        
        self.window_size_f, self.step_size_f = utils.calculate_spectrogram_params(
            self.params['window_size'],
            self.params['step_size_ratio'],
            self.rate
        )

        self.net_window_size_f = int(rate * self.params['net_window_size'])
        self.net_step_size_f = int(self.net_window_size_f * self.params['net_step_size_ratio'])

    def append_votes(self, stream_buffer):
        spectrogram_size_f = utils.calculate_coverage_size(
            stream_buffer.spectrogram.shape[1],
            self.step_size_f,
            self.window_size_f
        )

        ratio = stream_buffer.spectrogram.shape[1] / spectrogram_size_f

        net_window_size_pxl = int(np.ceil(self.net_window_size_f * ratio))
        net_step_size_pxl = int(np.ceil(self.net_step_size_f * ratio))

        cut_num_windows, cut_size_pxl = VoiceActivityDetector.calculate_cut_part_params(
            stream_buffer.spectrogram.shape[1],
            net_window_size_pxl,
            net_step_size_pxl
        )

        if cut_num_windows > 0:
            cut_part = stream_buffer.spectrogram[:, :cut_size_pxl, :]
            stream_buffer.spectrogram = stream_buffer.spectrogram[:, cut_size_pxl - net_window_size_pxl + net_step_size_pxl:, :]

            pxl_ls = np.arange(0, cut_part.shape[1], net_step_size_pxl)

            dataset = RealTimeVadDataset(cut_part, net_window_size_pxl, pxl_ls)
            data_loader = DataLoader(dataset, batch_size=len(pxl_ls), shuffle=False)
            pred_labels = self.predict(data_loader)

            mxl = 0
            for pred_label, pxl_l in zip(pred_labels, pxl_ls):
                pxl_r = pxl_l + net_step_size_pxl
                l = max(0, int(np.floor(pxl_l / ratio)))
                r = min(len(stream_buffer.speech_votes) - 1, int(np.ceil(pxl_r / ratio)))
                if l >= r:
                    continue
                mxl = l
                stream_buffer.total_votes[l] += 1
                stream_buffer.total_votes[r] -= 1
                if pred_label:
                    stream_buffer.speech_votes[l] += 1
                    stream_buffer.speech_votes[r] -= 1

            self.flush_predictions(stream_buffer, mxl)

    def append_spectrogram(self, stream_buffer):
        cut_num_windows, cut_size_f = VoiceActivityDetector.calculate_cut_part_params(
            len(stream_buffer.frames_buffer),
            self.window_size_f,
            self.step_size_f
        )
        if cut_num_windows > 0:
            cut_part = stream_buffer.frames_buffer[:cut_size_f]

            stream_buffer.frames_buffer = stream_buffer.frames_buffer[cut_size_f - self.window_size_f + self.step_size_f:]

            img = utils.build_spectrogram(
                cut_part,
                self.rate,
                self.params['n_filters'],
                self.params['window_size'],
                self.params['step_size_ratio'],
                last_prev_frame_signal=stream_buffer.last_prev_frame_signal
            )

            stream_buffer.last_prev_frame_signal = cut_part[-1]

            if stream_buffer.spectrogram is None:
                stream_buffer.spectrogram = img
            else:
                stream_buffer.spectrogram = np.concatenate((stream_buffer.spectrogram, img), axis=1)

    # should be called after 'setup(rate)'
    def append(self, added_frames, stream_buffer):
        stream_buffer.append(added_frames)
        self.append_spectrogram(stream_buffer)
        self.append_votes(stream_buffer)

    def query(self, stream_buffer):
        result = stream_buffer.labels
        stream_buffer.labels = np.array([], dtype=int)
        return result

    def flush_predictions(self, stream_buffer, c=np.inf):
        c = min(c, len(stream_buffer.total_votes))

        cut_total_votes, stream_buffer.total_votes = np.split(stream_buffer.total_votes, [c])
        cut_speech_votes, stream_buffer.speech_votes = np.split(stream_buffer.speech_votes, [c])

        cut_total_votes = np.cumsum(cut_total_votes)
        cut_speech_votes = np.cumsum(cut_speech_votes)

        stream_buffer.labels = np.append(stream_buffer.labels, (cut_speech_votes / (cut_total_votes + 1) > 0).astype(dtype=int))

        if len(cut_total_votes) > 0 and len(stream_buffer.total_votes) > 0:
            stream_buffer.total_votes[0] += cut_total_votes[-1]
        if len(cut_speech_votes) > 0 and len(stream_buffer.speech_votes) > 0:
            stream_buffer.speech_votes[0] += cut_speech_votes[-1]

    @staticmethod
    def calculate_cut_part_params(size, window_size, step_size):
        cut_num_windows = utils.calculate_num_windows(size, window_size, step_size)
        cut_size = 0
        while cut_num_windows > 0:
            cut_size = utils.calculate_coverage_size(cut_num_windows, step_size, window_size)
            if cut_size > size:
                cut_num_windows -= 1
            else:
                break
        return cut_num_windows, cut_size
