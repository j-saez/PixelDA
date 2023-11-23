import torch
import torch.nn as nn
from models.layers import GenConvBlock

class LineModClassifier(nn.Module):

    def __init__(self, in_chs: int, img_size: int, num_classes: int):
        """
        Mnist-M classifier network
        Inputs:
        Attributes:
        """
        super().__init__()

        self.private_layers = nn.Sequential(
            GenConvBlock(in_chs, out_chs=32, kernel_size=5, stride=1, padding=0, norm=True, use_relu=True),
            nn.MaxPool2d(kernel_size=2, stride=2),)

        self.shared_layers_1 = nn.Sequential(
            GenConvBlock(32, out_chs=48, kernel_size=5, stride=1, padding=0, norm=True, use_relu=True),
            nn.MaxPool2d(kernel_size=2, stride=2))

        self.shared_layers_2 = nn.Sequential(
            nn.Linear(48*2*(((img_size-4)//2)-4), 128),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.2))

        self.class_fcl = nn.Sequential(
            nn.Linear(128, num_classes),
            nn.Sigmoid())

        self.angle_fcl = nn.Sequential(
            nn.Linear(128, 1),
            nn.Tanh())
        return

    def forward(self, images: torch.tensor):
        """
        Performs the forward step for LineModClassifier 
        Inputs:
            >> images: (torch.tensor [Batch, CHS, IMG_H, IMG_W])
        Outputs:
            >> class_predictions: (torc.tensor [Batch, 11])
            >> angle_predictions: (torc.tensor [Batch, 1])
        """


        batch_size = images.shape[0]
        features = self.shared_layers_1(self.private_layers(images)).view(batch_size, -1)
        features = self.shared_layers_2(features)
        return self.class_fcl(features), self.angle_fcl(features)
