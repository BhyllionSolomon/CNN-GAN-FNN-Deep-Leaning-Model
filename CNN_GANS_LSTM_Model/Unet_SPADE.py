import torch
import torch.nn as nn
import torch.nn.functional as F

# SPADE layer
class SPADE(nn.Module):
    def __init__(self, norm_nc, segmap_nc):
        super(SPADE, self).__init__()
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)
        nhidden = 128
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(segmap_nc, nhidden, kernel_size=3, padding=1),
            nn.ReLU()
        )
        self.mlp_gamma = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)
        self.mlp_beta = nn.Conv2d(nhidden, norm_nc, kernel_size=3, padding=1)

    def forward(self, x, segmap):
        normalized = self.param_free_norm(x)
        segmap = F.interpolate(segmap, size=x.size()[2:], mode='nearest')
        actv = self.mlp_shared(segmap)
        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)
        return normalized * (1 + gamma) + beta

# Decoder block with SPADE
class DecoderBlockWithSPADE(nn.Module):
    def __init__(self, in_channels, out_channels, segmap_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.spade = SPADE(out_channels, segmap_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x, segmap):
        x = self.conv(x)
        x = self.spade(x, segmap)
        x = self.activation(x)
        return x
