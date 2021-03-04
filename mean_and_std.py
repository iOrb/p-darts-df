import logging

import torch
import torchvision.transforms as transforms
import torchvision.datasets as dset
import logging
import numpy as np


TRAIN_PATH = 'data/face_forensics/faces0/train'

def main():
    transform = transforms.Compose([
        transforms.Resize((160,160)),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor()
      ])

    train_data = dset.ImageFolder(root=TRAIN_PATH, transform=transform)
    dataloader = torch.utils.data.DataLoader(train_data, batch_size=4000, shuffle=False, num_workers=4)

    mean, std = online_mean_and_sd(dataloader)
    print(f'mean: {mean} \n std: {std}')

def online_mean_and_sd(loader):
    """Compute the mean and sd in an online fashion

        Var[x] = E[X^2] - E^2[X]
    """
    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in loader:

        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels

    return fst_moment, torch.sqrt(snd_moment - fst_moment ** 2)

if __name__ == '__main__':
    main()