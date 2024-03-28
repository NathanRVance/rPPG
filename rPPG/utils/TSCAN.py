"""Temporal Shift Convolutional Attention Network (TS-CAN).
Multi-Task Temporal Shift Attention Networks for On-Device Contactless Vitals Measurement
NeurIPS, 2020
Xin Liu, Josh Fromm, Shwetak Patel, Daniel McDuff
"""

import torch
import torch.nn as nn
import numpy as np

class Attention_mask(nn.Module):
    def __init__(self):
        super(Attention_mask, self).__init__()

    def forward(self, x):
        xsum = torch.sum(x, dim=2, keepdim=True)
        xsum = torch.sum(xsum, dim=3, keepdim=True)
        xshape = tuple(x.size())
        return x / xsum * xshape[2] * xshape[3] * 0.5


class TSM(nn.Module):
    def __init__(self, n_segment=10, fold_div=3):
        super(TSM, self).__init__()
        self.n_segment = n_segment
        self.fold_div = fold_div

    def forward(self, x):
        nt, c, h, w = x.size()
        n_batch = nt // self.n_segment
        x = x.view(n_batch, self.n_segment, c, h, w)
        fold = c // self.fold_div
        out = torch.zeros_like(x)
        out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
        out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
        out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift
        return out.view(nt, c, h, w)


class TSCAN_Metalayer(nn.Module):
    def __init__(self, frame_depth, in_channels, out_channels, kernel_size, pool_size, dropoutD, dropoutR):
        super(TSCAN_Metalayer, self).__init__()
        self.TSM1 = TSM(n_segment=frame_depth)
        self.TSM2 = TSM(n_segment=frame_depth)
        self.motion_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=(1, 1))
        self.motion_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size)
        self.apperance_conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=(1, 1))
        self.apperance_conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size)
        self.attn_mask = Attention_mask()
        self.apperance_att_conv1 = nn.Conv2d(out_channels, 1, kernel_size=1, padding=(0, 0))
        self.avg_pooling_1 = nn.AvgPool2d(pool_size)
        self.avg_pooling_2 = nn.AvgPool2d(pool_size)
        self.dropoutD = nn.Dropout(dropoutD)
        self.dropoutR = nn.Dropout(dropoutR)

    def forward(self, inputs):
        # Input: BN,in_channels,H,W
        # Output: BN,out_channels,H-2,W-2
        d, r = inputs[0], inputs[1]
        d = self.TSM1(d)
        d = torch.tanh(self.motion_conv1(d))
        d = self.TSM2(d)
        d = torch.tanh(self.motion_conv2(d))

        r = torch.tanh(self.apperance_conv1(r))
        r = torch.tanh(self.apperance_conv2(r))

        g = torch.sigmoid(self.apperance_att_conv1(r))
        g = self.attn_mask(g)
        g = d * g

        d = self.avg_pooling_1(g)
        d = self.dropoutD(d)

        r = self.avg_pooling_2(r)
        r = self.dropoutR(r)

        return torch.stack((d, r))

class TSCAN_reshape(nn.Module):
    def __init__(self):
        super(TSCAN_reshape, self).__init__()

    def forward(self, inputs):
        d, r = inputs[0], inputs[1]
        return d.view(d.size(0), -1)


from rPPG.utils import models
class TSCAN(models.Model):

    def __init__(self, config, **kwargs):
        """Definition of TS_CAN.
        """
        super(TSCAN, self).__init__(config)
        self.kernel_size = 3
        self.dropout_rate1 = 0.25
        self.dropout_rate2 = 0.5
        self.pool_size = (2,2)
        self.pool_null = (1,1)
        nb_filters1 = int(config.hidden_channels() // 2)
        nb_dense = config.hidden_channels() * 2
        frame_depth = config.fpc()
        assert config.frame_width() == config.frame_height()
        img_size = config.frame_width()

        def getDimLeft(skippedPooling):
            num = img_size
            for i in range(config.depth()):
                num -= 2
                if i not in skippedPooling:
                    num = int(num//2)
            return num

        skipIndices = []
        for i in list(range(1, config.depth(), 2)) + list(range(config.depth()-2 if config.depth()%2 == 0 else config.depth()-1, -1, -2)):
            if getDimLeft(skipIndices) > 0:
                break
            skipIndices.append(i)
        assert getDimLeft(skipIndices) > 0
        #print(sorted([s+1 for s in skipIndices]))

        pool = self.pool_null if 0 in skipIndices else self.pool_size
        layers = [TSCAN_Metalayer(frame_depth, len(config.channels()), nb_filters1, self.kernel_size, pool, self.dropout_rate1, self.dropout_rate1)]
        for i in range(1, config.depth()):
            in_channels = nb_filters1 if i == 1 else config.hidden_channels()
            dropout2nd = self.dropout_rate1 if i < config.depth()-1 else 0
            pool = self.pool_null if i in skipIndices else self.pool_size
            layers.append(TSCAN_Metalayer(frame_depth, in_channels, config.hidden_channels(), self.kernel_size, pool, self.dropout_rate1, dropout2nd))

        layers.append(TSCAN_reshape())
        
        # Dense layers
        num = getDimLeft(skipIndices)**2 * (nb_filters1 if config.depth() == 1 else config.hidden_channels())
        layers.append(nn.Linear(num, nb_dense, bias=True))
        layers.append(nn.Tanh())
        layers.append(nn.Dropout(self.dropout_rate2))
        layers.append(nn.Linear(nb_dense, 1, bias=True))
        self.forward_stream = nn.Sequential(*layers)

    def forward(self, inputs, params=None):
        # Reformat input: Input is BxCxTxHxW, but we want B*T,C,H,W
        inputs = torch.transpose(inputs, 1, 2)
        B,T,C,H,W = inputs.shape
        
        diff_input = torch.zeros_like(inputs)
        # Each index i <- (inputs[i] - inputs[i+1]) / (inputs[i] + inputs[i+1] + 1e-7)
        shifted = torch.zeros_like(inputs)
        shifted[:,:-1,...] = inputs[:,1:,...]
        diff_input = (inputs - shifted) / (inputs + shifted + 1e-7)
        diff_input = diff_input / torch.std(diff_input, dim=1, keepdim=True)
        diff_input[:,-1,...] = 0
        
        diff_input = diff_input.reshape(B*T, C, H, W)
        raw_input = inputs.reshape(B*T, C, H, W)

        out = self.forward_stream((diff_input, raw_input))

        out = out.view(B,T)
        return out
