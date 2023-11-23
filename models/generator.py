import torch
import torch.nn as nn
from models.layers import GenConvBlock

CHS_DIM = 1

class Generator(nn.Module):

    def __init__(self, in_img_chs: int, generated_img_chs: int, z_dim: int, img_size: int, n_res_blocks: int) -> None:
        """
        Implementation of the Discriminator proposed in the original paper
        Inputs:
            >> TODO
        Attributes:
            >> TODO
        """
        super().__init__()

        self.noise_projection = nn.Linear(z_dim, in_img_chs * img_size * img_size)
        self.conv1 = GenConvBlock(in_img_chs*2, 64, kernel_size=3, stride=1, padding=0, norm=True, use_relu=True)

        res_blocks = [] 
        for _ in range(n_res_blocks):
            res_blocks.append(ResidualBlock())

        self.res_blocks = nn.Sequential(*res_blocks)
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=generated_img_chs, kernel_size=3, stride=1, padding=2),
            nn.Tanh())
        return

    def forward(self, source_dom_imgs: torch.tensor, noise: torch.tensor) -> torch.tensor:
        """
        Forward step for the Discriminator
        Inputs:
            >> source_dom_imgs: (torch.tensor [batch, img_chs, img_size, img_size]) Source domain images.
            >> noise: (torch.tensor [batch, z_dim]) Noise vector.
        Outputs:
            >> fake_target_imgs: (torch.tensor [batch, img_chs, img_size, img_size]) Fake target domain images.
        """
        batch, img_chs, img_h, img_w = source_dom_imgs.shape
        noise = self.noise_projection(noise).view(batch, img_chs, img_h, img_w) # (batch, z_dim) -> (batch, img_chs, img_size, img_size)
        inputs = torch.cat(tensors=(source_dom_imgs, noise), dim=CHS_DIM) # (batch, img_chs, img_size, img_size) + (batch, img_chs, img_size, img_size) -> (batch, img_chs*2, img_size, img_size)
        return self.conv2(self.res_blocks(self.conv1(inputs)))

class ResidualBlock(nn.Module):

    def __init__(self,) -> None:
        """
        ResidualBlock for PixelDA generator
        Attributes:
            >> conv_block: (nn.Module)
        """
        super().__init__()

        self.conv_block = nn.Sequential(
            GenConvBlock(64, 64, kernel_size=3, stride=1, padding=1, norm=True, use_relu=True),
            GenConvBlock(64, 64, kernel_size=3, stride=1, padding=1, norm=True, use_relu=False),
        )
        return

    def forward(self, inputs: torch.tensor) -> torch.tensor:
        """
        Forward step for the ResidualBlock
        Inputs:
            >> inputs: (torch.tensor [batch, 64, input_size, input_size])
        Outputs:
            >> outpus: (torch.tensor [batch, 64, input_size - 4, input_size - 4])
        """
        return inputs + self.conv_block(inputs)


