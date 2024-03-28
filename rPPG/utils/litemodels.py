import torch
import torch.nn as nn
from rPPG.utils import losses
from rPPG.utils.models import Model

def preproc(x):
    if len(x.shape) == 5:
        # Transform [B,C,T,W,H] to [B,C,T]
        x = torch.mean(x, dim=(3,4))
    x = torch.squeeze(x, 1)
    return x

class LiteNet_v2(Model):
    def __init__(self, config, padding_mode='replicate', **kwargs):
        super(LiteNet_v2, self).__init__(config)
        channels = len(config.channels())
        relu = nn.ReLU() #nn.LeakyReLU()
        chan1 = 16
        kern1 = 5
        pad1 = kern1 // 2
        chan2 = 8
        kern2 = 3
        pad2 = kern2 // 2
        self.layers = nn.Sequential(
                nn.Conv1d(in_channels=channels, out_channels=chan1, kernel_size=kern1, stride=1, padding=pad1, padding_mode=padding_mode),
                nn.BatchNorm1d(chan1),
                relu,
                nn.Conv1d(in_channels=chan1, out_channels=chan1, kernel_size=kern1, stride=1, padding=pad1, padding_mode=padding_mode),
                nn.BatchNorm1d(chan1),
                relu,
                nn.Conv1d(in_channels=chan1, out_channels=chan2, kernel_size=kern2, stride=1, padding=pad2, padding_mode=padding_mode),
                nn.BatchNorm1d(chan2),
                relu,
                nn.Conv1d(in_channels=chan2, out_channels=chan2, kernel_size=kern2, stride=1, padding=pad2, padding_mode=padding_mode),
                nn.BatchNorm1d(chan2),
                relu,
                nn.Conv1d(in_channels=chan2, out_channels=chan2, kernel_size=kern2, stride=1, padding=pad2, padding_mode=padding_mode),
                nn.Conv1d(in_channels=chan2, out_channels=chan2, kernel_size=kern2, stride=1, padding=pad2, padding_mode=padding_mode),
                nn.Conv1d(in_channels=chan2, out_channels=chan2, kernel_size=kern2, stride=1, padding=pad2, padding_mode=padding_mode),
                nn.Conv1d(in_channels=chan2, out_channels=1, kernel_size=kern2, stride=1, padding=pad2, padding_mode=padding_mode),
                )

    def forward(self, x):
        ## Input should be of shape [B,num_waves,fpc]
        ## Output will be [B,fpc]
        x = self.layers(preproc(x))
        x = torch.flatten(x, start_dim=1)
        return x


class LiteNet_v6(Model):
    def __init__(self, config, padding_mode='replicate', **kwargs):
        super(LiteNet_v6, self).__init__(config)
        channels = len(config.channels())
        relu = nn.ReLU() #nn.LeakyReLU()
        chan1 = 64
        kern1 = 5
        pad1 = kern1 // 2
        chan2 = 32
        kern2 = 5
        pad2 = kern2 // 2
        self.layers = nn.Sequential(
                nn.Conv1d(in_channels=channels, out_channels=chan1, kernel_size=kern1, stride=1, padding=pad1, padding_mode=padding_mode),
                nn.BatchNorm1d(chan1),
                relu,
                nn.Conv1d(in_channels=chan1, out_channels=chan1, kernel_size=kern1, stride=1, padding=pad1, padding_mode=padding_mode),
                nn.BatchNorm1d(chan1),
                relu,
                nn.Conv1d(in_channels=chan1, out_channels=chan2, kernel_size=kern2, stride=1, padding=pad2, padding_mode=padding_mode),
                nn.BatchNorm1d(chan2),
                relu,
                nn.Conv1d(in_channels=chan2, out_channels=1, kernel_size=kern2, stride=1, padding=pad2, padding_mode=padding_mode),
                )

    def forward(self, x):
        ## Input should be of shape [B,num_waves,fpc]
        ## Output will be [B,fpc]
        x = self.layers(preproc(x))
        x = torch.flatten(x, start_dim=1)
        return x

