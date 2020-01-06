import torch
import PIL
import pickle
import numpy as np
from skimage import io


import torch.nn.functional as F
from tqdm import tqdm, tqdm_notebook
from PIL import Image
from pathlib import Path

from torchvision import transforms
from multiprocessing.pool import ThreadPool
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from matplotlib import colors, pyplot as plt
import typing

from sklearn.model_selection import train_test_split


RESCALE_SIZE = 80
DATA_MODES = ['train', 'val', 'test']
DEVICE = torch.device('cuda')
# DEVICE = torch.device('cpu')


class VadDataset(Dataset):

    MEAN = np.array([0.485, 0.456, 0.406])
    STD = np.array([0.229, 0.224, 0.225])

    def __init__(self, files: typing.List[Path], mode):
        super().__init__()
        self.files = sorted(files)
        assert mode in DATA_MODES
        self.mode = mode
        self.len = len(files)
        self.label_encoder = LabelEncoder()

        if self.mode != 'test':
            self.labels = [path.parent.name for path in self.files]
            self.label_encoder.fit(self.labels)

    def __len__(self):
        return self.len

    def load_sample(self, path):
        image = Image.open(path)
        image.load()
        return image

    def __getitem__(self, index):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(VadDataset.MEAN, VadDataset.STD)
        ])
        x = self.load_sample(self.files[index])
        print(np.array(x).shape)
        x = np.array(x.resize((RESCALE_SIZE, RESCALE_SIZE)), dtype='float32')[:, :, :3] / 255
        x = transform(x)
        if self.mode == 'test':
            return x
        else:
            label_id = self.label_encoder.transform([self.labels[index]]).item()
            return x, label_id


def imshow(inp, title=None, plt_ax=plt):
    inp = inp.numpy().transpose((1, 2, 0))
    inp = VadDataset.STD * inp + VadDataset.MEAN
    inp = np.clip(inp, 0, 1)
    print()
    print(inp.shape, np.min(inp), np.max(inp))
    plt_ax.imshow(inp)
    if title is not None:
        plt_ax.set_title(title)
    plt_ax.grid(False)


class VadNet(nn.Module):

    def __init__(self):
        # starting image shape = RESCALE_SIZE x RESCALE_SIZE x 3
        # float32
        # [0, 1]
        # normalized tensor
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=2),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=2),
            nn.ReLU()
        )
        self.lin1 = nn.Linear(5184, 60)
        self.drop = nn.Dropout(p=0.25)
        self.lin2 = nn.Linear(60, 2)
        self.out = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = x.view(x.size(0), -1)
        x = self.lin1(x)
        x = self.drop(x)
        x = self.lin2(x)
        x = self.out(x)
        # print(x)
        return x


def train_epoch(train_loader, model, optimizer, loss_f):
    model.train()

    loss = 0.0
    samples_count = 0
    accuracy = 0

    for inputs, targets in train_loader:
        inputs = inputs.to(device=DEVICE, dtype=torch.float)
        targets = targets.to(device=DEVICE, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_f(outputs, targets)
        loss.backward()
        optimizer.step()
        train_preds = torch.argmax(outputs, 1)

        loss += loss.item() * inputs.size(0)

        accuracy += torch.sum(train_preds == targets.data).item()
        samples_count += inputs.size(0)

    print(f'train: loss = {loss}, accuracy = {accuracy}, samples_count = {samples_count}')

    loss /= samples_count
    accuracy /= samples_count

    return loss, accuracy


def validate_epoch(val_loader, model, loss_f):
    model.eval()

    loss = 0.0
    samples_count = 0
    accuracy = 0

    for inputs, targets in val_loader:
        inputs = inputs.to(device=DEVICE, dtype=torch.float)
        targets = targets.to(device=DEVICE, dtype=torch.long)

        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            loss = loss_f(outputs, targets)
            val_preds = torch.argmax(outputs, 1)

        loss += loss.item() * inputs.size(0)
        accuracy += torch.sum(val_preds == targets.data).item()
        samples_count += inputs.size(0)

    print(f'validation: loss = {loss}, accuracy = {accuracy}, samples_count = {samples_count}')

    loss /= samples_count
    accuracy /= samples_count

    return loss, accuracy


def train(train_dataset, val_dataset, model, epochs, batch_size):
    train_loader = DataLoader(train_dataset, batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters())
    loss_f = nn.CrossEntropyLoss()

    history = []

    for _ in tqdm(range(epochs)):
        train_loss, train_accuracy = train_epoch(train_loader, model, optimizer, loss_f)
        val_loss, val_accuracy = validate_epoch(val_loader, model, loss_f)
        history.append((train_loss, train_accuracy, val_loss, val_accuracy))

    return history


# def predict(model, test_loader):
#     with torch.no_grad():
#         for inputs in test_loader:
#             inputs = inputs.to(DEVICE)
#             model.eval()
#             outputs = model(inputs).cpu()
#

if __name__ == '__main__':
    train_on_gpu = torch.cuda.is_available()

    if not train_on_gpu:
        print('CUDA is not available.  Training on CPU ...')
    else:
        print('CUDA is available!  Training on GPU ...')

    print(f'PIL version = {PIL.PILLOW_VERSION}')

    DATASET_DIR = Path('dataset')
    dataset_paths = sorted(list(DATASET_DIR.rglob('*.png')))

    print(dataset_paths)

    labels = [path.parent.name for path in dataset_paths]

    X_train, X_rem, y_train, y_rem = train_test_split(dataset_paths, labels, test_size=0.3, shuffle=True)
    X_val, X_test, y_val, y_test = train_test_split(X_rem, y_rem, test_size=0.1, shuffle=True)

    print(len(X_train), len(y_train))
    print(len(X_val), len(y_val))
    print(len(X_test), len(y_test))

    train_dataset = VadDataset(X_train, mode='train')
    val_dataset = VadDataset(X_val, mode='val')
    test_dataset = VadDataset(X_test, mode='test')

    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(12, 12))
    for fig_x in ax.flatten():
        random_characters = int(np.random.uniform(0, val_dataset.len))
        im_val, label = val_dataset[random_characters]
        img_label = " ".join(map(lambda x: x.capitalize(), val_dataset.label_encoder.inverse_transform([label])[0].split('_')))
        imshow(im_val.data.cpu(), title=img_label, plt_ax=fig_x)

    plt.show()

    vad_cnn = VadNet().to(DEVICE)
    print(vad_cnn)

    history = train(train_dataset, val_dataset, vad_cnn, 10, batch_size=40)

    print(history)

    train_loss, train_accuracy, val_loss, val_accuracy = zip(*history)

    plt.figure(figsize=(15, 9))
    plt.legend(loc='best')

    plt.subplot(211)
    plt.plot(train_loss, label="train_loss")
    plt.plot(val_loss, label="val_loss")
    plt.xlabel("epochs")
    plt.ylabel("loss")

    plt.subplot(212)
    plt.plot(train_accuracy, label="train_accuracy")
    plt.plot(val_accuracy, label="val_accuracy")
    plt.xlabel("epochs")
    plt.ylabel("accuracy")

    plt.legend()
    plt.show()

    print(train_accuracy)
    print()
    print(val_accuracy)

    torch.save(vad_cnn.state_dict(), 'saved_models/model')
