import sys
sys.path.append('../../')

import modules.torch_vnd as nn_vnd
import modules.torch_vdo as nn_vdo
import modules.torch_vnd_eval as nn_vnd_eval

from torch.nn import init

from torch import nn
import torch.nn.functional as F
import torch

cfgs = {
    '11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    '11-wide': [96, 'M', 192, 'M', 384, 384, 'M', 768, 768, 'M', 768, 768, 'M'],
    '11-wide-2x': [128, 'M', 256, 'M', 512, 512, 'M', 1024, 1024, 'M', 1024, 1024, 'M'],
}


class VGG_VND(nn.Module):
    def __init__(self, output_shape, batch_norm, config, NUM_DIV, FREEZE_PART, PI):
        super(VGG_VND, self).__init__()
        self.NUM_DIV = NUM_DIV
        self.FREEZE_PART = FREEZE_PART
        self.PI = PI

        self.batch_norm = batch_norm
        self.features = self.make_layers(cfgs[config], batch_norm)
        # self.dense1 = nn.Linear(768, 256)
        # self.dense2 = nn.Linear(256, output_shape)
        self.dense1 = nn_vdo.LinearVDO(768, 256, ard_init=-1.)
        self.dense2 = nn_vdo.LinearVDO(256, output_shape, ard_init=-1.)

        if self.batch_norm:
            self.bn1 = nn.BatchNorm1d(256)

    def make_layers(self, cfg, batch_norm):
        layers = []
        in_channels = 3
        for idx, v in enumerate(cfg):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if idx == 0:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                else:
                    conv2d = nn_vnd.Conv2dVND(in_channels, v, kernel_size=3, padding=1,
                                                NUM_DIV=self.NUM_DIV, FREEZE_PART = self.FREEZE_PART, PI=self.PI, ard_init=-1.)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)

        if self.batch_norm:
            out = F.relu(self.bn1(self.dense1(out)), inplace=True)
        else:
            out = F.relu(self.dense1(out), inplace=True)

        out = self.dense2(out)

        return out

class VGG_VND_eval(nn.Module):
    def __init__(self, output_shape, batch_norm, config, NUM_DIV, FREEZE_PART, PI, multiple):
        super(VGG_VND_eval, self).__init__()
        self.NUM_DIV = NUM_DIV
        self.FREEZE_PART = FREEZE_PART
        self.PI = PI

        indense = cfgs[config][-2] * multiple

        self.features = self.make_layers(cfgs[config], batch_norm)

        self.dense1 = nn_vdo.LinearVDO(768, 256, ard_init=-1.)
        self.dense2 = nn_vdo.LinearVDO(256, output_shape, ard_init=-8.)

        if batch_norm:
            self.bn1 = nn.BatchNorm1d(256)


    def make_layers(self, cfg, batch_norm):
        layers = []
        in_channels = 3
        for idx, v in enumerate(cfg):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                if idx == 0:
                    conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                else:
                    conv2d = nn_vnd_eval.Conv2dVND_eval(in_channels, v, kernel_size=3, padding=1,
                                                       NUM_DIV=self.NUM_DIV, FREEZE_PART = self.FREEZE_PART, PI=self.PI, ard_init=-1.)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = F.relu(self.bn1(self.dense1(out)), inplace=True)
        out = self.dense2(out)

        return out

