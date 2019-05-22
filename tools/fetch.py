#!/usr/bin/env python
"""
Gets trained model and warms it up (i.e. compiles and dumps corresponding prediction functions)
"""
import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cakechat.utils.env import init_keras

init_keras()

from cakechat.dialog_model.factory import get_trained_model
from cakechat.utils.logger import get_tools_logger
from cakechat.utils.w2v.model import get_w2v_model

_logger = get_tools_logger(__file__)


def parse_args():
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        '-m',
        '--model',
        action='store',
        choices=['default', 'reverse', 'w2v', 'all'],
        help='Fetch models from s3 to disk',
        default='all')
    args = argparser.parse_args()

    return args


if __name__ == '__main__':
    args = parse_args()

    if args.model in {'default', 'all'}:
        get_trained_model(fetch_from_s3=True)

    if args.model in {'reverse', 'all'}:
        get_trained_model(fetch_from_s3=True, is_reverse_model=True)

    if args.model in {'w2v', 'all'}:
        get_w2v_model(fetch_from_s3=True)
