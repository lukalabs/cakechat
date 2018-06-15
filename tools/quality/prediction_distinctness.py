from __future__ import print_function
import os
import sys
import argparse

from six.moves import xrange

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from cakechat.utils.env import init_theano_env

init_theano_env()

from cakechat.dialog_model.model import get_nn_model
from cakechat.dialog_model.model_utils import get_model_full_path
from cakechat.dialog_model.quality import calculate_response_ngram_distinctness
from cakechat.utils.dataset_loader import load_datasets, load_questions_set
from cakechat.utils.text_processing import get_index_to_token_path, load_index_to_item, get_index_to_condition_path
from cakechat.utils.logger import get_tools_logger
from cakechat.config import BASE_CORPUS_NAME, PREDICTION_MODES, PREDICTION_MODE_FOR_TESTS

_logger = get_tools_logger(__file__)


def log_distinct_metrics(nn_model, x, condition_ids=None, samples_num=1, ngram_lengths=(1, 2, 3)):
    for ngram_length in ngram_lengths:
        scores = [
            calculate_response_ngram_distinctness(x, nn_model, ngram_len=ngram_length, condition_ids=condition_ids)
            for _ in xrange(samples_num)
        ]
        scores_mean = np.mean(scores)
        scores_std = np.std(scores)
        result = 'distinct {}-gram = {:.5f}'.format(ngram_length, scores_mean)
        if samples_num > 1:
            result += ' (std: {:.5f})'.format(scores_std)
        print(result)


def load_model(model_path=None, tokens_index_path=None, conditions_index_path=None):
    if model_path is None:
        model_path = get_model_full_path()
    if tokens_index_path is None:
        tokens_index_path = get_index_to_token_path(BASE_CORPUS_NAME)
    if conditions_index_path is None:
        conditions_index_path = get_index_to_condition_path(BASE_CORPUS_NAME)

    index_to_token = load_index_to_item(tokens_index_path)
    index_to_condition = load_index_to_item(conditions_index_path)
    nn_model, model_exists = get_nn_model(index_to_token, index_to_condition, nn_model_path=model_path)

    if not model_exists:
        raise ValueError('Couldn\'t find model: "{}".'.format(model_path))

    return nn_model


def evaluate_distinctness(args):
    if args.sample_size > 1 and PREDICTION_MODE_FOR_TESTS == PREDICTION_MODES.beamsearch:
        _logger.waring('Using sample_size > 1 is meaningless with prediction_mode=\'beamsearch\' because there\'s no '
                       'randomness in the prediction. Use sample_size=1 instead.')

    nn_model = load_model(args.model, args.tokens_index, args.conditions_index)

    if args.validation_only:
        validation = load_questions_set(nn_model.token_to_index)
        validation_set_name = 'context free questions'
    else:
        train, _, validation, train_subset, defined_condition_subset = load_datasets(
            nn_model.token_to_index, nn_model.condition_to_index)

        validation_set_name = 'validation set without conditions'

        _logger.info('Evaluating distinctness for train subset without conditions')
        log_distinct_metrics(nn_model, train_subset.x, samples_num=args.sample_size)

        _logger.info('Evaluating distinctness for train subset with conditions')
        log_distinct_metrics(nn_model, train_subset.x, train_subset.condition_ids, samples_num=args.sample_size)

        _logger.info('Evaluating distinctness for defined-conditions-subset without conditions')
        log_distinct_metrics(nn_model, defined_condition_subset.x, samples_num=args.sample_size)

        _logger.info('Evaluating distinctness for defined-conditions-subset with conditions')
        log_distinct_metrics(
            nn_model, defined_condition_subset.x, defined_condition_subset.condition_ids, samples_num=args.sample_size)

    _logger.info('Evaluating distinctness for {}'.format(validation_set_name))
    log_distinct_metrics(nn_model, validation.x, samples_num=args.sample_size)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        '-m',
        '--model',
        action='store',
        default=None,
        help='Path to the file with your model. '
        'Be careful, model parameters are inferred from the config, not from the filename')

    argparser.add_argument(
        '-t',
        '--tokens_index',
        action='store',
        default=None,
        help='Path to the json file with index_to_token dictionary.')

    argparser.add_argument(
        '-c',
        '--conditions_index',
        action='store',
        default=None,
        help='Path to the json file with index_to_condition dictionary.')

    argparser.add_argument(
        '-s', '--sample_size', action='store', default=1, type=int, help='Number of samples to average over')

    argparser.add_argument(
        '-v',
        '--validation_only',
        action='store_true',
        help='Evaluate on the validation set only (useful if you are impatient)')

    args = argparser.parse_args()

    evaluate_distinctness(args)
