import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


# Reference
# https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights.
    Parameters:
        net (network)   -- network to be initialized
        init_type (str) -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        init_gain (float)    -- scaling factor for normal, xavier and orthogonal.
    We use 'normal' in the original pix2pix and CycleGAN paper. But xavier and kaiming might
    work better for some applications. Feel free to try yourself.
    """
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)  # apply the initialization function <init_func>


def init_net(net, init_type='normal', init_gain=0.02, gpu_ids=[]):
    """Initialize a network: 1. register CPU/GPU device (with multi-GPU support); 2. initialize the network weights
    Parameters:
        net (network)      -- the network to be initialized
        init_type (str)    -- the name of an initialization method: normal | xavier | kaiming | orthogonal
        gain (float)       -- scaling factor for normal, xavier and orthogonal.
        gpu_ids (int list) -- which GPUs the network runs on: e.g., 0,1,2
    Return an initialized network.
    """
    if len(gpu_ids) > 0:
        assert(torch.cuda.is_available())
        net.to(gpu_ids[0])
        net = torch.nn.DataParallel(net, gpu_ids)  # multi-GPUs
    init_weights(net, init_type, init_gain=init_gain)
    return net


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, norm='instance', activation='relu'):
        super(ConvBlock, self).__init__()
        padding = (kernel_size - 1) // 2
        self.layers = []
        self.layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride))
        if norm == 'instance':
            self.layers.append(nn.InstanceNorm2d(out_channels))

        if activation == 'leaky':
            self.layers.append(nn.LeakyReLU(0.2, True))
        elif activation == 'relu':
            self.layers.append(nn.ReLU(True))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class DeconvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, norm='instance', activation='relu'):
        super(DeconvBlock, self).__init__()
        padding = (kernel_size - 1) // 2
        self.layers = []
        self.layers.append(nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, stride=stride, output_padding=0))
        if norm == 'instance':
            self.layers.append(nn.InstanceNorm2d(out_channels))

        if activation == 'leaky':
            self.layers.append(nn.LeakyReLU(0.2, True))
        elif activation == 'relu':
            self.layers.append(nn.ReLU(True))

        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.layers(x)


class ResBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, stride=1):
        super(ResBlock, self).__init__()
        padding = (kernel_size - 1) // 2

        self.conv = nn.Sequential(
            nn.ReflectionPad2d(padding=padding),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(True),
            nn.ReflectionPad2d(padding=padding),
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernel_size, stride=stride),
            nn.InstanceNorm2d(in_channels)
        )

    def forward(self, x):
        return x + self.conv(x)


class CycleGAN(nn.Module):
    def __init__(self):
        super(CycleGAN, self).__init__()
        self.layers = nn.ModuleList()

        self.encoder1 = ConvBlock(3, 64, kernel_size=7, stride=1)
        self.encoder2 = ConvBlock(64, 128, kernel_size=4, stride=2)
        self.encoder3 = ConvBlock(128, 256, kernel_size=4, stride=2)

        res = []
        for i in range(9):
            res.append(ResBlock(256, kernel_size=3, stride=1))
        self.res = nn.Sequential(*res)

        self.decoder1 = DeconvBlock(256, 128, kernel_size=4, stride=2)
        self.decoder2 = DeconvBlock(128, 64, kernel_size=4, stride=2)
        self.decoder3 = ConvBlock(64, 3, kernel_size=7, stride=1, norm='none', activation='none')

    def forward(self, x):
        out = self.encoder1(x)
        out = self.encoder2(out)
        out = self.encoder3(out)

        out = self.res(out)

        out = self.decoder1(out)
        out = self.decoder2(out)
        out = self.decoder3(out)

        # out = self.layers(x)
        out = torch.tanh(out)

        return out


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        layer_list = ['C-64', 'C-128', 'C-256', 'C-512']

        self.layers = nn.ModuleList()

        in_channels = 3
        for idx, _ in enumerate(layer_list):
            block, out_channels = _.split('-')
            out_channels = int(out_channels)
            self.layers.append(ConvBlock(in_channels, out_channels, kernel_size=4, stride=2, activation='leaky'))
            in_channels = out_channels
        self.layers.append(nn.Conv2d(512, 1, kernel_size=4, padding=1, bias=False))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, x):
        out = self.layers(x)
        return out
