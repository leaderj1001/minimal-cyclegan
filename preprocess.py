import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

import numpy as np
import os
from PIL import Image


class Monet2Photo(Dataset):
    def __init__(self, dataset='monet2photo', mode='train'):
        super(Monet2Photo, self).__init__()

        self.transforms = transforms.Compose([
            transforms.Resize(286),
            transforms.RandomCrop(256),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])

        self.base_dir = './dataset/{}'.format(dataset)

        self.A_path = os.path.join(self.base_dir, '{}A'.format(mode))
        self.A_list = os.listdir(self.A_path)

        self.B_path = os.path.join(self.base_dir, '{}B'.format(mode))
        self.B_list = os.listdir(self.B_path)

    def __getitem__(self, index):
        img_A = Image.open(os.path.join(self.A_path, self.A_list[index]))
        img_B = Image.open(os.path.join(self.B_path, self.B_list[index]))

        if self.transforms is not None:
            img_A = self.transforms(img_A)
            img_B = self.transforms(img_B)

        data = {
            'A': img_A,
            'pathA': os.path.join(self.A_path, self.A_list[index]),
            'B': img_B,
            'pathB': os.path.join(self.B_path, self.B_list[index])
        }

        return data

    def __len__(self):
        if len(self.A_list) < len(self.B_list):
            return len(self.A_list)
        return len(self.B_list)


def load_data(args):
    train_data = Monet2Photo()
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    test_data = Monet2Photo(mode='test')
    test_loader = DataLoader(test_data, batch_size=args.batch_size, num_workers=args.num_workers)

    return train_loader, test_loader


def main():

    from config import load_args
    args = load_args()
    load_data(args)


# if __name__ == '__main__':
#     main()