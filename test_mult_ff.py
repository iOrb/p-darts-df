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
parser.add_argument('--report_freq', type=float, default=50, help='report frequency')
parser.add_argument('--epochs', type=int, default=600, help='num of training epochs')
parser.add_argument('--drop_path_prob', type=float, default=0.3, help='drop path probability')
parser.add_argument('--init_channels', type=int, default=36, help='num of init channels')
parser.add_argument('--layers', type=int, default=20, help='total number of layers')
parser.add_argument('--auxiliary', action='store_true', default=False, help='use auxiliary tower')
parser.add_argument('--auxiliary_weight', type=float, default=0.4, help='weight for auxiliary loss')
parser.add_argument('--cutout', action='store_true', default=False, help='use cutout')
parser.add_argument('--cutout_length', type=int, default=16, help='cutout length')
parser.add_argument('--center_crop', action='store_true', default=False, help='use center crop')
parser.add_argument('--arch', type=str, default='PDARTS', help='which architecture to use')
parser.add_argument('--grad_clip', type=float, default=5, help='gradient clipping')
parser.add_argument('--data_dir', type=str, default='data/', help='data dir')
parser.add_argument('--model', type=str, default='models/model.pth.tar', help='data dir')
parser.add_argument('--eval_mode', type=str, default='multiclass', help='could be "multiclass" (p, df, fs, f2f, nt) or "binary" (real, fake)')
parser.add_argument('--note', type=str, default='try', help='note for this run')

args, unparsed = parser.parse_known_args()

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
    format=log_format, datefmt='%m/%d %I:%M:%S %p')

NUM_CLASSES = 5

def main():
    if not torch.cuda.is_available():
        logging.info('No GPU device available')
        sys.exit(1)
    cudnn.benchmark = True
    cudnn.enabled=True
    logging.info("args = %s", args)
    logging.info("unparsed args = %s", unparsed)

    genotype = eval("genotypes.%s" % args.arch)
    print('---------Genotype---------')
    logging.info(genotype)
    print('--------------------------')

    model = Network(args.init_channels, NUM_CLASSES, args.layers, args.auxiliary, genotype)

    model = torch.nn.DataParallel(model)
    model = model.cuda()
    logging.info("param size = %fMB", utils.count_parameters_in_MB(model))

    model.load_state_dict(torch.load(args.model)['state_dict'])
    epoch = torch.load(args.model)['epoch']

    criterion = nn.CrossEntropyLoss()
    criterion = criterion.cuda()

    model.module.drop_path_prob = args.drop_path_prob * epoch / args.epochs
    model.drop_path_prob = args.drop_path_prob * epoch / args.epochs

    test_queue = get_data(args, args.batch_size)

    if 'multiclass' in args.eval_mode:
        test_obj, test_acc_mult, test_acc_bi, test_sens_bi, test_spec_bi  = infer_mult(test_queue, model, criterion)
        logging.info('Test_acc_mult: %f', test_acc_mult)
        logging.info('Test_acc_bi: %f', test_acc_bi)
        logging.info('Test_sensitivity: %f', test_sens_bi)
        logging.info('Test_specificity: %f', test_spec_bi)
        logging.info('Test_loss: %f', test_obj)
    elif 'binary' in args.eval_mode:
        test_obj, test_acc_bi, test_sens_bi, test_spec_bi = infer_binary(test_queue, model, criterion)
        logging.info('Test_acc_bi: %f', test_acc_bi)
        logging.info('Test_sensitivity: %f', test_sens_bi)
        logging.info('Test_specificity: %f', test_spec_bi)
        logging.info('Test_loss: %f', test_obj)
    else:
        print('Eval mode -',args.eval_mode,'- NOT supported')


def infer_mult(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    acc_mult = utils.AvgrageMeter()
    acc_bi = utils.AvgrageMeter()
    sens_bi = utils.AvgrageMeter()
    spec_bi = utils.AvgrageMeter()
    model.eval()

    logging.info('\t%-12s %-14s %-12s %-12s %-22s %-22s' % ("Test_Step", "Objs", "Acc.mult", "Acc.bi", "Sensitivity.bi", "Specificity.bi"))
    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits, _ = model(input)
            loss = criterion(logits, target)

        n = input.size(0)
        objs.update(loss.data.item(), n)

        # ===
        acc_mu, = utils.accuracy(output=logits, target=target)
        acc_b, sens_b, spec_b = utils.confusion_matrix_mult_as_binary(output=logits, target=target, target_type_mult=True)
        # ===

        acc_mult.update(acc_mu, n)
        acc_bi.update(acc_b, n)
        sens_bi.update(sens_b, n)
        spec_bi.update(spec_b, n)

        if step % args.report_freq == 0:
            logging.info('\t%-12d %-14e %-12f %-12f %-22f %-22f' % (step, objs.avg, acc_mult.avg, acc_bi.avg, sens_bi.avg, spec_bi.avg))

    return objs.avg, acc_mult.avg, acc_bi.avg, sens_bi.avg, spec_bi.avg
    sys.exit(1)


def infer_binary(valid_queue, model, criterion):
    objs = utils.AvgrageMeter()
    acc_ = utils.AvgrageMeter()
    sens_ = utils.AvgrageMeter()
    spec_ = utils.AvgrageMeter()
    model.eval()

    logging.info('\t%-12s %-14s %-10s %-13s %-13s' % ("Test_Step","Objs","Acc","Sensitivity","Specificity"))
    for step, (input, target) in enumerate(valid_queue):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        with torch.no_grad():
            logits, _ = model(input)
            loss = criterion(logits, target)

        n = input.size(0)
        objs.update(loss.data.item(), n)

        # ===
        acc, sens, spec = utils.confusion_matrix_mult_as_binary(output=logits,target=target, target_type_mult=False)
        # ===

        acc_.update(acc, n)
        sens_.update(sens, n)
        spec_.update(spec, n)

        if step % args.report_freq == 0:
            logging.info('\t%-12d %-14e %-10f %-13f %-13f' % (step, objs.avg, acc_.avg, sens_.avg, spec_.avg))

    return objs.avg, acc_.avg, sens_.avg, spec_.avg
    sys.exit(1)

def get_data(args, batch_size):
    # mean, std = utils.online_mean_and_sd(os.path.join(args.data_dir, 'val'))
    mean, std = [0.5978, 0.4296, 0.3926], [0.2297, 0.1851, 0.1784]
    _ , test_transform  = utils._data_transform_ff(args, size=160, mean=mean, std=std)

    test_data = dset.ImageFolder(root=os.path.join(args.data_dir, 'test'), transform=test_transform)

    test_queue = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=args.workers, drop_last=True)

    return test_queue


if __name__ == '__main__':
    start_time = time.time()
    main() 
    end_time = time.time()
    duration = end_time - start_time
    logging.info('Eval time: %ds.', duration)
