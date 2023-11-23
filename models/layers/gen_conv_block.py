import torch
import torch.nn as nn

class GenConvBlock(nn.Module):
    def __init__(self, in_chs: int, out_chs: int, kernel_size: int, stride: int, padding: int, norm: bool, use_relu: bool) -> None:
        """
        TODO
        """
        super().__init__()

        # As there will be normalization, bias is not needed.
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, bias=not norm)
        self.batch_norm = nn.BatchNorm2d(out_chs)
        self.lrelu = nn.LeakyReLU(0.2)
        self.apply_norm = norm
        self.use_relu = use_relu 

    def forward(self, inputs: torch.tensor):
        out = self.conv(inputs)
        if self.apply_norm: out = self.batch_norm(out)
        if self.use_relu:   out = self.lrelu(out)
        return out
