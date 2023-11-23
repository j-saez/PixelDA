import torch
import torch.nn as nn

class MNISTMClassifier(nn.Module):

    def __init__(self, in_chs: int, img_size: int, num_classes: int):
        """
        Mnist-M classifier network
        Inputs:
        Attributes:
        """
        super().__init__()

        self.private_layers = nn.Sequential(
            ConvBlock(in_chs, out_chs=32, kernel_size=5, stride=1, padding=0, use_relu=True),
            nn.MaxPool2d(kernel_size=2, stride=2),)

        self.shared_layers = nn.Sequential(
            ConvBlock(32, out_chs=48, kernel_size=5, stride=1, padding=0, use_relu=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Linear(img_size-8, 100),
            nn.LeakyReLU(0.2),
            nn.Linear(100, 100),
            nn.LeakyReLU(0.2),
            nn.Linear(100, 10),
            nn.Sigmoid(),)
        return

    def forward(self, images: torch.tensor):
        """
        Performs the forward step for MNISTMClassifier
        Inputs:
            >> images: (torch.tensor [Batch, CHS, IMG_H, IMG_W])
        Outputs:
            >> predictions: (torc.tensor [Batch, 10])
        """
        return self.shared_layers(self.private_layers(images))

if __name__ == "__main__":
    inputs = torch.rand(4,3,28,28)
    model = MNISTMClassifier(in_chs=3, img_size=28, num_classes=20)
    outputs = model(inputs)
    print(f'resnet34 outputs shape = {outputs.shape}')
