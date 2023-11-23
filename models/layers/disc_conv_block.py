import torch
import torch.nn as nn

class DiscConvBlock(nn.Module):
    """
    TODO
    """
    def __init__(self, in_chs: int, out_chs: int, kernel_size: int, stride: int, padding: int, norm: bool) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_chs, out_chs, kernel_size, stride, padding, bias=not norm)
        self.batch_norm = nn.InstanceNorm2d(out_chs)
        self.lrelu = nn.LeakyReLU(0.2)
        self.apply_norm = norm
        return

    def forward(self, inputs: torch.tensor):
        out = self.conv(inputs)
        if self.apply_norm: out = self.batch_norm(out)
        out = self.lrelu(out)
        return out
