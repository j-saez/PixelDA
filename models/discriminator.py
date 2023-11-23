import torch
import torch.nn as nn
from models.layers import DiscConvBlock

class Discriminator(nn.Module):

    def __init__(self, img_chs: int, img_size: int) -> None:
        """
        Implementation of the Generator proposed in the original paper
        Inputs:
            >>
        Attributes:
            >>
        """
        super().__init__()

        input_size = img_size
        last_conv_layer_out_size = 0
        stride = [1,2,2,2,2]
        kernel_size = [3,3,3,3,1]
        for i in range(5):
            last_conv_layer_out_size = int((input_size - kernel_size[i])/stride[i] + 1)
            input_size = last_conv_layer_out_size 

        self.conv1 = DiscConvBlock(img_chs, 64, kernel_size=3, stride=1, padding=0, norm=True)
        self.conv2 = DiscConvBlock(64, 128, kernel_size=3, stride=2, padding=0, norm=True)
        self.conv3 = DiscConvBlock(128, 256, kernel_size=3, stride=2, padding=0, norm=True)
        self.conv4 = DiscConvBlock(256, 512, kernel_size=3, stride=2, padding=0, norm=True)
        self.conv5 = DiscConvBlock(512, 1024, kernel_size=1, stride=2, padding=0, norm=True)
        self.conv5 = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=1, stride=2, padding=0),
            nn.LeakyReLU(0.2))

        self.fcl = nn.Sequential(
                nn.Linear(1024 * last_conv_layer_out_size * last_conv_layer_out_size, 1),
                nn.Sigmoid())

        self.dropout = nn.Dropout(p=0.1) # Keep prob of 90%

        return

    def forward(self, input_images: torch.tensor) -> torch.tensor:
        """
        Forward step for the Discriminator 
        Inputs:
            >> input_images: (torch.tensor [batch, img_chs, img_size, img_size])
        Outputs:
            >> out: (torch.tensor [batch, 1]) Values between [0,1] for each one of the input images (TODO: 0 = fake, 1 = real?)
        """
        batch_size = input_images.shape[0]
        std_deviation = 0.2

        conv1_out = self.conv1(input_images)
        conv1_out = conv1_out + (torch.randn(size=conv1_out.size(), device=conv1_out.device) * std_deviation)
        conv1_out = self.dropout(conv1_out)

        conv2_out = self.conv2(conv1_out)
        conv2_out = conv2_out + (torch.randn(size=conv2_out.size(), device=conv1_out.device) * std_deviation) 
        conv2_out = self.dropout(conv2_out)

        conv3_out = self.conv3(conv2_out)
        conv3_out = conv3_out + (torch.randn(size=conv3_out.size(), device=conv1_out.device) * std_deviation) 
        conv3_out = self.dropout(conv3_out)

        conv4_out = self.conv4(conv3_out)
        conv4_out = conv4_out + (torch.randn(size=conv4_out.size(), device=conv1_out.device) * std_deviation) 
        conv4_out = self.dropout(conv4_out)

        conv5_out = self.conv5(conv4_out)
        conv5_out = conv5_out + (torch.rand(size=conv5_out.size(), device=conv5_out.device) * std_deviation)
        conv5_out = self.dropout(conv5_out)

        return self.fcl(conv5_out.view(batch_size,-1)) 

def get_random_noise_for_discriminator(noise_size: tuple):
    """
    Gets random noise to be injected in every discriminator layer,
    drawn from a zero centered Gaussian with stddev 0.2.

    Inputs:
        >> noise_size: (tuple) Size of the output noise.
    Outputs:
        >> noise: (torch.tensor) Random noise from a zero centered gausssian with stddev 0.2
    """
    std_deviation=0.2
    noise = torch.randn(size=noise_size) * std_deviation
    noise = torch.clamp(noise, -1, 1)
    return noise
