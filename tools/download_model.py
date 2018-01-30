#!/usr/bin/env python
"""
Gets trained model and warms it up (i.e. compiles and dumps corresponding prediction functions)
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cakechat.utils.env import init_theano_env

init_theano_env()

from cakechat.dialog_model.factory import get_trained_model
from cakechat.utils.logger import get_tools_logger

_logger = get_tools_logger(__file__)

if __name__ == '__main__':
    _logger.info('Fetching and pre-compiling pre-trained model...')
    get_trained_model(fetch_from_s3=True)
    _logger.info('Successfully resolved and compiled model.')
    _logger.info('Fetching and pre-compiling additional reverse-model for MMI reranking...')
    get_trained_model(fetch_from_s3=True, reverse=True)
    _logger.info('Successfully resolved and compiled reverse-model.')
