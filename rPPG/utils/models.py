import torch
import torch.nn as nn
from rPPG.utils import losses
import numpy as np

class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        self.config = config
        lossObject = losses.Loss(config)
        self.loss = lossObject.loss

class CNN3D(Model):
    # Adapted from https://github.com/ZitongYu/PhysNet
    def __init__(self, config, padding_mode='replicate', **kwargs):
        super(CNN3D, self).__init__(config)
        channels = len(config.channels())
        tk=config.tk()

        t_pad = int(tk / 2)

        self.conv1 = nn.Conv3d(in_channels=channels, out_channels=32, kernel_size=(1,5,5), padding=(0,2,2), padding_mode=padding_mode)
        self.bn1 = nn.BatchNorm3d(32)
        self.max_pool1 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.conv2 = nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(tk,3,3), padding=(t_pad,1,1), padding_mode=padding_mode)
        self.bn2 = nn.BatchNorm3d(64)

        self.conv3 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(tk,3,3), padding=(t_pad,1,1), padding_mode=padding_mode)
        self.bn3 = nn.BatchNorm3d(64)
        self.max_pool2 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.conv4 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(tk,3,3), padding=(t_pad,1,1), padding_mode=padding_mode)
        self.bn4 = nn.BatchNorm3d(64)

        self.conv5 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(tk,3,3), padding=(t_pad,1,1), padding_mode=padding_mode)
        self.bn5 = nn.BatchNorm3d(64)
        self.max_pool3 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.conv6 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(tk,3,3), padding=(t_pad,1,1), padding_mode=padding_mode)
        self.bn6 = nn.BatchNorm3d(64)

        self.conv7 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(tk,3,3), padding=(t_pad,1,1), padding_mode=padding_mode)
        self.bn7 = nn.BatchNorm3d(64)
        self.max_pool4 = nn.MaxPool3d(kernel_size=(1,2,2), stride=(1,2,2))
        self.conv8 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(tk,3,3), padding=(t_pad,1,1), padding_mode=padding_mode)
        self.bn8 = nn.BatchNorm3d(64)

        self.conv9 = nn.Conv3d(in_channels=64, out_channels=64, kernel_size=(tk,3,3), padding=(t_pad,1,1), padding_mode=padding_mode)
        self.bn9 = nn.BatchNorm3d(64)

        self.avg_pool1 = nn.AvgPool3d(kernel_size=(1,4,4), stride=(1,2,2))
        self.conv10 = nn.Conv3d(in_channels=64, out_channels=1, kernel_size=1)

        self.drop3d = nn.Dropout3d(config.dropout())

        self.forward_stream = nn.Sequential(
            self.conv1, self.bn1, nn.ReLU(), self.max_pool1,
            self.conv2, self.bn2, nn.ReLU(),
            self.conv3, self.bn3, nn.ReLU(), self.drop3d, self.max_pool2,
            self.conv4, self.bn4, nn.ReLU(),
            self.conv5, self.bn5, nn.ReLU(), self.drop3d, self.max_pool3,
            self.conv6, self.bn6, nn.ReLU(),
            self.conv7, self.bn7, nn.ReLU(), self.drop3d, self.max_pool4,
            self.conv8, self.bn8, nn.ReLU(),
            self.conv9, self.bn9, nn.ReLU(), self.drop3d, self.avg_pool1,
            self.conv10
        )

    def forward(self, x):
        x = self.forward_stream(x)
        x = torch.flatten(x, start_dim=1, end_dim=4)
        return x

class Flex(Model):
    def __init__(self, config, **kwargs):
        super(Flex, self).__init__(config)
        
        # We need to get from frame_width x frame_height down to 1x1 over the course of depth-1 layers
        assert np.log2(config.frame_width()).is_integer() and np.log2(config.frame_height()).is_integer()
        narrowBins = [[int(np.log2(dim)//(config.depth()-1)) for _ in range(config.depth()-1)] for dim in [config.frame_width(), config.frame_height()]]
        for bins, dim in zip(narrowBins, [config.frame_width(), config.frame_height()]):
            deficit = np.log2(dim) - sum(bins)
            i = len(bins)-1
            while deficit > 0 and i >= 0:
                bins[i] += 1
                deficit -= 1
                i -= 2
            i += 1
            if i < 0:
                i += 2
            while deficit > 0 and i < len(bins):
                bins[i] += 1
                deficit -= 1
                i += 2
            assert sum(bins) == np.log2(dim)
        narrowBins = np.swapaxes(np.array(narrowBins), 0, 1)

        outChan1 = 32 if config.hidden_channels() > 32 and config.depth() > 2 else 1 if config.depth() == 1 else config.hidden_channels()
        layers = [nn.Conv3d(in_channels=len(config.channels()), out_channels=outChan1, kernel_size=(1,5,5), padding=(0,2,2), padding_mode=config.padding_mode()),
                  nn.BatchNorm3d(outChan1),
                  nn.ReLU()]
        def getPoolKernel(bins):
            return [1] + [2**b for b in bins]
        if np.sum(narrowBins[0]) > 0:
            poolType = torch.nn.AvgPool3d if config.depth() == 2 else torch.nn.MaxPool3d
            layers.append(poolType(kernel_size=getPoolKernel(narrowBins[0])))
        for i in range(1, config.depth()-1):
            dilation=1
            if config.flex_dilation() and i % 2 == 0:
                dilation = int(2**(i/2-1))
            in_channels = outChan1 if i == 1 else config.hidden_channels()
            out_channels = config.out_channels() if i == config.depth()-1 else config.hidden_channels()
            layers.append(nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=(config.tk(),3,3), dilation=(dilation,1,1), padding=((config.tk()//2)*dilation,1,1), padding_mode=config.padding_mode()))
            layers.append(nn.BatchNorm3d(out_channels))
            layers.append(nn.ReLU())
            if i % 2 == 0:
                layers.append(nn.Dropout3d(config.dropout()))
            if np.sum(narrowBins[i]) > 0:
                poolType = torch.nn.AvgPool3d if i == config.depth()-2 else torch.nn.MaxPool3d
                layers.append(poolType(kernel_size=getPoolKernel(narrowBins[i])))
        # Finally, tack on our last conv layer
        in_channels = outChan1 if config.depth() == 2 else config.hidden_channels()
        layers.append(nn.Conv3d(in_channels=in_channels, out_channels=config.out_channels(), kernel_size=1))

        self.forward_stream = nn.Sequential(*layers)

    def forward(self, x):
        #print(f'Input: {x.shape}')
        ## Input should be of shape [B,C,T,W,H]
        ## Output will be [B,T]
        x = self.forward_stream(x)
        x = torch.squeeze(x, 4)
        x = torch.squeeze(x, 3)
        x = torch.squeeze(x, 1)
        if self.config.out_channels() > 1: # Convert [B,C,T] to [B,T,C]
            x = torch.transpose(x, 1, 2)
        #print(f'Output: {x.shape}')
        return x

class NRNet(Model):
    def __init__(self, config, **kwargs):
        super(NRNet, self).__init__(config)
        self.head = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=128, kernel_size=5, stride=(1,1), padding=(0,2)), nn.BatchNorm2d(128), nn.LeakyReLU(),
                nn.Conv2d(in_channels=128, out_channels=128, kernel_size=5, stride=(1,1), padding=(0,2)), nn.BatchNorm2d(128), nn.LeakyReLU(),
                nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=(1,1), padding=(0,0)), nn.BatchNorm2d(1), nn.LeakyReLU(),
                nn.Flatten(start_dim=2, end_dim=3),
                nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1), nn.BatchNorm1d(64), nn.LeakyReLU(),
                )
        self.splits = nn.ModuleList([
            nn.Sequential(nn.Conv1d(in_channels=64, out_channels=32, kernel_size=K, stride=1, padding=P), nn.BatchNorm1d(32), nn.LeakyReLU())
            for K, P in [(3, 1), (5, 2), (7, 3)]])
        self.middle = nn.Sequential(
                nn.Linear(in_features=32, out_features=1),
                nn.Dropout(p=0.5)
                )
        self.tail = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1), nn.BatchNorm1d(64), nn.LeakyReLU(),
                nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),nn.BatchNorm1d(64), nn.LeakyReLU(),
                nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.Conv1d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
                nn.Tanh()
                )

    def forward(self, x):
        ## Input should be of shape [B,1,num_waves=9,fpc]
        ## Output will be [B,fpc]
        xave = x[:,:,-1,:] # Final signal should be average of rest
        x = self.head(x)
        x = torch.stack([s(x) for s in self.splits])
        x = torch.mean(x, dim=0)
        x = torch.transpose(x, 1, 2) # linear layer expects this
        x = self.middle(x)
        x = torch.transpose(x, 1, 2) # linear layer expects this
        x = torch.add(x, xave)
        x = self.tail(x)
        x = torch.flatten(x, start_dim=1)
        return x

class NRNet_simple(Model):
    def __init__(self, config, **kwargs):
        super(NRNet_simple, self).__init__(config)
        self.layers = nn.Sequential(
                nn.Conv1d(in_channels=config.num_waves(), out_channels=128, kernel_size=5, stride=1, padding=2), nn.BatchNorm1d(128), nn.LeakyReLU(),
                nn.Conv1d(in_channels=128, out_channels=128, kernel_size=5, stride=1, padding=2), nn.BatchNorm1d(128), nn.LeakyReLU(),
                nn.Conv1d(in_channels=128, out_channels=64, kernel_size=3, stride=1, padding=1), nn.BatchNorm1d(64), nn.LeakyReLU(),
                nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1), nn.BatchNorm1d(64), nn.LeakyReLU(),
                nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.Conv1d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
                nn.Conv1d(in_channels=64, out_channels=1, kernel_size=3, stride=1, padding=1),
                nn.Tanh()
                )

    def forward(self, x):
        ## Input should be of shape [B,1,num_waves=4,fpc] (due to NRNet)
        ## Input must be transformed to be of shape [B,num_waves=4,fpc]
        x = torch.flatten(x, start_dim=1, end_dim=2)
        ## Output will be [B,fpc]
        x = self.layers(x)
        x = torch.flatten(x, start_dim=1)
        return x
