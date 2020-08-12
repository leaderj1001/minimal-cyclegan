import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

from config import load_args
from preprocess import load_data
from model import CycleGAN, Discriminator, init_net

import itertools
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


# Reference
# https://discuss.pytorch.org/t/simple-way-to-inverse-transform-normalization/4821
class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
            # The normalize code -> t.sub_(m).div_(s)
        return tensor


class ToNumpy(object):
    def __call__(self, data):
        return data.cpu().detach().numpy().transpose(0, 2, 3, 1)


def set_requires_grad(nets, requires_grad=False):
    """Set requies_grad=Fasle for all the networks to avoid unnecessary computations
    Parameters:
        nets (network list)   -- a list of networks
        requires_grad (bool)  -- whether the networks require gradients or not
    """
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


def train(train_loader, generatorA2B, generatorB2A, discriminatorA, discriminatorB, optimizerG, optimizerD, cycle, gan, identity, args, epoch):
    generatorA2B.train()
    generatorB2A.train()
    discriminatorA.train()
    discriminatorB.train()

    for i, data in enumerate(train_loader):
        dataA, dataB, pathA, pathB = data['A'], data['B'], data['pathA'], data['pathB']

        if args.cuda:
            dataA, dataB = dataA.cuda(), dataB.cuda()

        genB = generatorA2B(dataA)
        resA = generatorB2A(genB)

        genA = generatorB2A(dataB)
        resB = generatorA2B(genA)

        set_requires_grad([discriminatorA, discriminatorB], True)
        optimizerD.zero_grad()

        disA_real = discriminatorA(dataA)
        disA_fake = discriminatorA(genA.detach())

        loss_D_A_real = gan(disA_real, torch.ones_like(disA_real))
        loss_D_A_fake = gan(disA_fake, torch.zeros_like(disA_fake))
        loss_D_A = 0.5 * (loss_D_A_real + loss_D_A_fake)

        disB_real = discriminatorB(dataB)
        disB_fake = discriminatorB(genB.detach())

        loss_D_B_real = gan(disB_real, torch.ones_like(disB_real))
        loss_D_B_fake = gan(disB_fake, torch.zeros_like(disB_fake))
        loss_D_B = 0.5 * (loss_D_B_real + loss_D_B_fake)

        loss_D = loss_D_A + loss_D_B
        loss_D.backward()
        optimizerD.step()

        set_requires_grad([discriminatorA, discriminatorB], False)
        optimizerG.zero_grad()

        identity_b = generatorA2B(dataB)
        identity_a = generatorB2A(dataA)

        loss_identity_a = identity(identity_a, dataA)
        loss_identity_b = identity(identity_b, dataB)

        identity_loss = 5e-1 * (10 * loss_identity_a + 10 * loss_identity_b)

        fake_a = discriminatorA(genA)
        fake_b = discriminatorB(genB)

        loss_G_a = gan(fake_b, torch.ones_like(fake_b))
        loss_G_b = gan(fake_a, torch.ones_like(fake_a))
        cycle_loss = 10 * cycle(dataA, resA) + 10 * cycle(dataB, resB)

        loss_G = (loss_G_a + loss_G_b) + cycle_loss + identity_loss

        loss_G.backward()
        optimizerG.step()

        print('[Epoch: {0:4d}], Loss Discriminator: {1:.4f}, Loss Generator: {2:.4f}'.format(epoch, loss_D, loss_G))

        if i % args.print_intervals == 0:
            if not os.path.isdir('E:/cyclegan/checkpoints'):
                os.mkdir('E:/cyclegan/checkpoints')

            torch.save({
                'generatorA': generatorA2B.state_dict(),
                'generatorB': generatorB2A.state_dict(),
                'discriminatorA': discriminatorA.state_dict(),
                'discriminatorB': discriminatorB.state_dict(),
                'optimizerG': optimizerG.state_dict(),
                'optimizerD': optimizerD.state_dict(),
                'start_epoch': epoch,
            }, 'E:/cyclegan/checkpoints/model_{}_{}.pth'.format(epoch, i))


def evaluation(test_loader, generatorA2B, generatorB2A, args):
    generatorA2B.eval()
    generatorB2A.eval()

    transform_inv = transforms.Compose([UnNormalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)), ToNumpy()])

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            dataA, dataB, pathA, pathB = data['A'], data['B'], data['pathA'], data['pathB']

            if args.cuda:
                dataA, dataB = dataA.cuda(), dataB.cuda()

            outB = generatorA2B(dataA)
            outA = generatorB2A(dataB)

            resB = generatorA2B(outA)
            resA = generatorB2A(outB)

            dataA = transform_inv(dataA)
            dataB = transform_inv(dataB)
            outA = transform_inv(outA)
            outB = transform_inv(outB)
            resA = transform_inv(resA)
            resB = transform_inv(resB)

            if args.results:
                if not os.path.isdir('results'):
                    os.mkdir('results')

                for j in range(1, dataA.shape[0]):
                    name = args.batch_size * i + j
                    plt.imsave('results/{0:4d}_input_a.png'.format(name), dataA[j - 1, :, :, :])
                    plt.imsave('results/{0:4d}_input_b.png'.format(name), dataB[j - 1, :, :, :])
                    plt.imsave('results/{0:4d}_out_a.png'.format(name), outA[j - 1, :, :, :])
                    plt.imsave('results/{0:4d}_out_b.png'.format(name), outB[j - 1, :, :, :])
                    plt.imsave('results/{0:4d}_recon_a.png'.format(name), resA[j - 1, :, :, :])
                    plt.imsave('results/{0:4d}_recon_b.png'.format(name), resB[j - 1, :, :, :])


def main(args):
    train_loader, test_loader = load_data(args)

    GeneratorA2B = CycleGAN()
    GeneratorB2A = CycleGAN()

    DiscriminatorA = Discriminator()
    DiscriminatorB = Discriminator()

    if args.cuda:
        GeneratorA2B = GeneratorA2B.cuda()
        GeneratorB2A = GeneratorB2A.cuda()

        DiscriminatorA = DiscriminatorA.cuda()
        DiscriminatorB = DiscriminatorB.cuda()

    optimizerG = optim.Adam(itertools.chain(GeneratorA2B.parameters(), GeneratorB2A.parameters()), lr=args.lr, betas=(0.5, 0.999))
    optimizerD = optim.Adam(itertools.chain(DiscriminatorA.parameters(), DiscriminatorB.parameters()), lr=args.lr, betas=(0.5, 0.999))

    if args.training:
        path = 'E:/cyclegan/checkpoints/model_{}_{}.pth'.format(285, 200)

        checkpoint = torch.load(path)
        GeneratorA2B.load_state_dict(checkpoint['generatorA'])
        GeneratorB2A.load_state_dict(checkpoint['generatorB'])
        DiscriminatorA.load_state_dict(checkpoint['discriminatorA'])
        DiscriminatorB.load_state_dict(checkpoint['discriminatorB'])
        optimizerG.load_state_dict(checkpoint['optimizerG'])
        optimizerD.load_state_dict(checkpoint['optimizerD'])

        start_epoch = 285
    else:
        init_net(GeneratorA2B, init_type='normal', init_gain=0.02, gpu_ids=[0])
        init_net(GeneratorB2A, init_type='normal', init_gain=0.02, gpu_ids=[0])

        init_net(DiscriminatorA, init_type='normal', init_gain=0.02, gpu_ids=[0])
        init_net(DiscriminatorB, init_type='normal', init_gain=0.02, gpu_ids=[0])
        start_epoch = 1

    if args.evaluation:
        evaluation(test_loader, GeneratorA2B, GeneratorB2A, args)
    else:
        cycle = nn.L1Loss()
        gan = nn.BCEWithLogitsLoss()
        identity = nn.L1Loss()

        for epoch in range(start_epoch, args.epochs):
            train(train_loader, GeneratorA2B, GeneratorB2A, DiscriminatorA, DiscriminatorB, optimizerG, optimizerD, cycle, gan, identity, args, epoch)
        evaluation(test_loader, GeneratorA2B, GeneratorB2A, args)


if __name__ == '__main__':
    args = load_args()
    main(args)
