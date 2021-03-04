import os
import numpy as np
import torch
import shutil
import torchvision.transforms as transforms
from torch.autograd import Variable
import torchvision.datasets as dset
import logging
from sklearn.metrics import confusion_matrix


def _data_transform_ff(args, size=160, mean=None, std=None):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        # transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
        transforms.RandomPerspective(distortion_scale=0.5, p=0.3),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    try:
        if args.cutout:
            train_transform.transforms.append(Cutout(args.cutout_length))
    except:
        pass

    valid_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return train_transform, valid_transform


def _data_transform_search(args, size=160, mean=None, std=None):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    try:
        if args.cutout:
            train_transform.transforms.append(Cutout(args.cutout_length))
    except:
        pass

    valid_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return train_transform, valid_transform


def _data_transform_mult(args, size=160, mean=None, std=None):
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomResizedCrop(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333), interpolation=2),
        transforms.ColorJitter(
            brightness=0.1,
            contrast=0.1,
            saturation=0.1,
            hue=0.05),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    try:
        if args.cutout:
            train_transform.transforms.append(Cutout(args.cutout_length))
    except:
        pass

    valid_transform = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    return train_transform, valid_transform


def _data_transforms_cifar10(args):
    CIFAR_MEAN = [0.49139968, 0.48215827, 0.44653124]
    CIFAR_STD = [0.24703233, 0.24348505, 0.26158768]

    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    if args.cutout:
        train_transform.transforms.append(Cutout(args.cutout_length))

    valid_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    return train_transform, valid_transform


def online_mean_and_sd(data_path):
    """Compute the mean and sd in an online fashion

            Var[x] = E[X^2] - E^2[X]
        """
    transform = transforms.Compose([
        transforms.Resize(164),
        transforms.ToTensor()
    ])
    valid_data = dset.ImageFolder(root=data_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(valid_data, batch_size=4000, shuffle=False, num_workers=4)

    cnt = 0
    fst_moment = torch.empty(3)
    snd_moment = torch.empty(3)

    for images, _ in dataloader:
        b, c, h, w = images.shape
        nb_pixels = b * h * w
        sum_ = torch.sum(images, dim=[0, 2, 3])
        sum_of_square = torch.sum(images ** 2, dim=[0, 2, 3])
        fst_moment = (cnt * fst_moment + sum_) / (cnt + nb_pixels)
        snd_moment = (cnt * snd_moment + sum_of_square) / (cnt + nb_pixels)

        cnt += nb_pixels
    mean = fst_moment.numpy()
    std = torch.sqrt(snd_moment - fst_moment ** 2).numpy()
    logging.info(f'MEAN: {mean}')
    logging.info(f'STD: {std}')
    return mean, std


class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy_binary_data(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    pred[pred < 4] = 0
    pred[pred == 4] = 1
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def accuracy_mult_as_binary(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    pred[pred < 4] = 0;
    pred[pred == 4] = 1
    target[target < 4] = 0;
    target[target == 4] = 1
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def confusion_matrix_binary(output, target):
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t().data.cpu().numpy()[0]
    target = target.data.cpu().numpy()
    pred[pred == 0] = 2
    pred[pred == 1] = 0
    pred[pred == 2] = 1
    target[target == 0] = 2
    target[target == 1] = 0
    target[target == 2] = 1
    tn, fp, fn, tp = conf_matrix(target, pred)

    return tn, fp, fn, tp


def confusion_matrix_mult_as_binary(output, target, target_type_mult=True):
    _, pred = output.topk(1, 1, True, True)
    pred = pred.t().data.cpu().numpy()[0]
    target = target.data.cpu().numpy()
    pred[pred < 4] = 1
    pred[pred == 4] = 0
    if target_type_mult:
        target[target < 4] = 1
        target[target == 4] = 0
    else:
        target[target == 0] = 2
        target[target == 1] = 0
        target[target == 2] = 1
    tn, fp, fn, tp = confusion_matrix(target, pred).ravel()
    acc = (tp + tn) / (tp + fp + tn + fn)
    sensitivity = 0 if tp == 0 else tp / (tp + fn)
    specificity = 0 if tn == 0 else tn / (tn + fp)

    return acc * 100, sensitivity * 100, specificity * 100


def conf_matrix(target, pred):
    assert len(pred) == len(target)
    tn, fp, fn, tp = 0, 0, 0, 0
    for t, p in zip(target, pred):
        if p == 1 and t == 1:
            tp += 1
        elif p == 0 and t == 0:
            tn += 1
        elif p == 1 and t == 0:
            fp += 1
        elif p == 0 and t == 1:
            fn += 1
    return tn, fp, fn, tp


class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


def count_parameters_in_MB(model):
    return np.sum(np.prod(v.size()) for name, v in model.named_parameters() if "auxiliary" not in name) / 1e6


def save_checkpoint(state, is_best, save):
    filename = os.path.join(save, 'checkpoint.pth.tar')
    torch.save(state, filename)
    if is_best:
        best_filename = os.path.join(save, 'model_best.pth.tar')
        shutil.copyfile(filename, best_filename)


def save(model, model_path):
    torch.save(model.state_dict(), model_path)


def load(model, model_path):
    model.load_state_dict(torch.load(model_path))


def drop_path(x, drop_prob):
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = Variable(torch.cuda.FloatTensor(x.size(0), 1, 1, 1).bernoulli_(keep_prob))
        x.div_(keep_prob)
        x.mul_(mask)
    return x


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.mkdir(path)
    print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.mkdir(os.path.join(path, 'scripts'))
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean

    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)
