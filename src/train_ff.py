import copy
import os
import sys
import time
import glob
import numpy as np
import torch
import utils
import genotypes
import logging
import argparse
import torch.nn as nn
import torch.utils
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
#
from matplotlib import pyplot as plt

from model import NetworkDF as Network


parser = argparse.ArgumentParser()
parser.add_argument('--workers', type=int, default=4, help='number of workers')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--learning_rate', type=float, default=0.1, help='init learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
parser.add_argument('--weight_decay', type=float, default=3e-5, help='weight decay')
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--center_crop', action='store_true', default=False, help='use center crop')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--save', type=str, default='/tmp/checkpoints/', help='experiment name')
parser.add_argument('--seed', type=int, default=0, help='random seed')
parser.add_argument('--arch', type=str, default='PDARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--data_dir', type=str, default='data/', help='data dir')
parser.add_argument('--note', type=str, default='try', help='note for this run')

args, unparsed = parser.parse_known_args()

args.save = '{}eval-{}-{}'.format(args.save, args.note, time.strftime("%Y%m%d-%H%M%S"))
utils.create_exp_dir(args.save, scripts_to_save=glob.glob('*.py'))

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')
fh = logging.FileHandler(os.path.join(args.save, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

NUM_CLASSES = 2
CLASSES = {1:'original',
           0:'deepfake'}

def main():
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)
    np.random.seed(args.seed)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled=True
    torch.cuda.manual_seed(args.seed)
    logging.info("args = %s", args)
    logging.info("unparsed args = %s", unparsed)
    num_gpus = torch.cuda.device_count()
    
    genotype = eval("genotypes.%s" % args.arch)
    print('---------Genotype---------')
    logging.info(genotype)
    print('--------------------------')

    # Infer optimal batch_size
    # new_batch_size = infer_batch_size(args, args.batch_size, genotype)

    new_batch_size = args.batch_size
    model = Network(args.init_channels, NUM_CLASSES, args.layers, args.auxiliary, genotype)

    model = torch.nn.DataParallel(model)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()
    optimizer = torch.optim.SGD(
        model.parameters(),
        args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay
        )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, float(args.epochs))

    # new_batch_size = args.batch_size
    train_queue, valid_queue = get_data(args, new_batch_size)

    best_acc = 0.0
    train_accs = []
    test_accs = []
    epoch_hist = []
    for epoch in range(args.epochs):
        scheduler.step()
        logging.info('Epoch: %d lr %e', epoch, scheduler.get_lr()[0])
        model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        model.drop_path_prob = args.drop_path_prob * epoch / args.epochs
        start_time = time.time()
        train_acc, train_obj = train(train_queue, model, criterion, optimizer)
        logging.info('Train_acc: %f', train_acc)
        logging.info('Train_loss: %f', train_obj)

        valid_acc, valid_obj = infer(valid_queue, model, criterion)
        is_best = False
        if valid_acc >= best_acc:
            best_acc = valid_acc
            is_best = True
        logging.info('Valid_acc: %f', valid_acc)
        logging.info('Valid_loss: %f', valid_obj)
        end_time = time.time()
        duration = end_time - start_time
        print('Epoch time: %ds.' % duration )
        train_accs.append(train_acc)
        test_accs.append(valid_acc)
        epoch_hist.append(epoch)
        # utils.save(model.module, os.path.join(args.save, 'weights.pt'))
        utils.save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_acc_top1': best_acc,
            'optimizer': optimizer.state_dict(),
        }, is_best, args.save)
    # save_results(epoch_hist,train_accs,test_accs, args)


def train(train_queue, model, criterion, optimizer):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.train()

    for step, (input, target) in enumerate(train_queue):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)



        optimizer.zero_grad()
        try:
            logits, logits_aux = model(input)
        except:
            logits, logits_aux = model(input)
            print(f'Error with model input size: {input.size()}')
            sys.exit(1)
        loss = criterion(logits, target)
        if args.auxiliary:
            loss_aux = criterion(logits_aux, target)
            loss += args.auxiliary_weight*loss_aux
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        prec1, = utils.accuracy(logits, target)
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('Train Step: %03d Objs: %e Acc: %f', step, objs.avg, top1.avg)

    return top1.avg, objs.avg


def infer(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    top1 = utils.AvgrageMeter()
    model.eval()

    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits, _ = model(input)
            loss = criterion(logits, target)

        prec1, = utils.accuracy(logits, target)
        n = input.size(0)
        objs.update(loss.data.item(), n)
        top1.update(prec1.data.item(), n)

        if step % args.report_freq == 0:
            logging.info('Valid Step: %03d Objs: %e Acc: %f', step, objs.avg, top1.avg)

    return top1.avg, objs.avg


def get_data(args, batch_size):
    # mean, std = utils.online_mean_and_sd(os.path.join(args.data_dir, 'val'))
    mean, std = [0.5978, 0.4296, 0.3926], [0.2297, 0.1851, 0.1784]
    train_transform, val_transform  = utils._data_transform_ff(args, size=160, mean=mean, std=std)

    train_data = dset.ImageFolder(root=os.path.join(args.data_dir, 'train'), transform=train_transform)
    val_data = dset.ImageFolder(root=os.path.join(args.data_dir, 'val'), transform=val_transform)

    train_queue = torch.utils.data.DataLoader(
            train_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=args.workers, drop_last=True)
    valid_queue = torch.utils.data.DataLoader(
            val_data, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=args.workers, drop_last=True)

    return train_queue, valid_queue


if __name__ == '__main__':
    start_time = time.time()
    main() 
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Eval time: %ds.', duration)
