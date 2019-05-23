import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cakechat.utils.env import run_horovod_train


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-g', '--gpu-ids', action='store', nargs='+', required=True)
    argparser.add_argument('-s', '--train-subset-size', action='store', type=int)
    args = argparser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    train_cmd = 'python tools/train.py'
    if args.train_subset_size:
        train_cmd += ' -s {}'.format(args.train_subset_size)

    run_horovod_train(train_cmd, args.gpu_ids)
