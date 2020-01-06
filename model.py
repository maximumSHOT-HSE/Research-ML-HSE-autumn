import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import utils
from cnn import CNN


class VoiceActivityDetector:

    def __init__(self, params=None):
        self.params = params
        self.net = CNN()

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
    DEVICE = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    def to_device(self):
        self.net = self.net.to(VoiceActivityDetector.DEVICE)

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

    def validate_epoch(self, val_loader, optimizer, loss_f):
        self.net.eval()

        cum_loss = 0.0
        samples_count = 0
        accuracy = 0

        for inputs, targets in val_loader:
            inputs = inputs.to(device=VoiceActivityDetector.DEVICE, dtype=torch.float)
            targets = targets.to(device=VoiceActivityDetector.DEVICE, dtype=torch.long)

            with torch.set_grad_enabled(False):
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
        self.to_device()

        optimizer = torch.optim.Adam(self.net.parameters())
        loss_f = nn.CrossEntropyLoss()

        history = []
        best_val_loss = np.inf

        for _ in tqdm(range(epochs)):
            train_loss, train_accuracy = self.train_epoch(train_loader, optimizer, loss_f)
            val_loss, val_accuracy = self.validate_epoch(val_loader, optimizer, loss_f)
            history.append((train_loss, train_accuracy, val_loss, val_accuracy))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                utils.save_model(sessions_dir, session_id, self)

        return history

    def predict(self, inputs):
        inputs = inputs.to(device=VoiceActivityDetector.DEVICE, dtype=torch.float)
        with torch.set_grad_enabled(False):
            outputs = self.net(inputs)
            print(outputs)
            return torch.argmax(outputs, 1)
