from PIL import Image
from torch.utils.data import Dataset

import model


class RealTimeVadDataset(Dataset):

    def __init__(self, spectrogram, sample_pxl_width, pxl_ls):
        super().__init__()
        self.spectrogram = spectrogram
        self.sample_pxl_width = sample_pxl_width
        self.pxl_ls = pxl_ls
        self.len = len(pxl_ls)

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        pxl_l = self.pxl_ls[index]
        x = self.spectrogram[:, pxl_l: pxl_l + self.sample_pxl_width, :]
        x = Image.fromarray(x)
        x = model.VoiceActivityDetector.from_picture_to_tensor(x)
        return x
