import torch.nn as nn


class CNN(nn.Module):

    IMG_SIZE = (64, 16)

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=4, kernel_size=3),
            nn.ReLU()
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=3, stride=(1, 2)),
            nn.ReLU()
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=(1, 2)),
            nn.ReLU()
        )

        self.lin1 = nn.Linear(2240, 4)
        self.drop = nn.Dropout(p=0.25)
        self.lin2 = nn.Linear(4, 2)
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
        return x
