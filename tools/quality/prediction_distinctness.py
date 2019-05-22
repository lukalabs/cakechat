import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np

from cakechat.utils.env import init_cuda_env

init_cuda_env()

from cakechat.dialog_model.factory import get_trained_model
from cakechat.dialog_model.quality import calculate_response_ngram_distinctness
from cakechat.utils.dataset_loader import load_datasets, load_questions_set
from cakechat.utils.logger import get_tools_logger
from cakechat.config import PREDICTION_MODES, PREDICTION_MODE_FOR_TESTS

_logger = get_tools_logger(__file__)


def log_distinct_metrics(nn_model, x, condition_ids=None, samples_num=1, ngram_lengths=(1, 2, 3)):
    for ngram_length in ngram_lengths:
        scores = [
            calculate_response_ngram_distinctness(x, nn_model, ngram_len=ngram_length, condition_ids=condition_ids)
            for _ in range(samples_num)
        ]
        scores_mean = np.mean(scores)
        scores_std = np.std(scores)
        result = 'distinct {}-gram = {:.5f}'.format(ngram_length, scores_mean)
        if samples_num > 1:
            result += ' (std: {:.5f})'.format(scores_std)
        print(result)


def evaluate_distinctness(args):
    if args.sample_size > 1 and PREDICTION_MODE_FOR_TESTS == PREDICTION_MODES.beamsearch:
        _logger.waring('Using sample_size > 1 is meaningless with prediction_mode=\'beamsearch\' because there\'s no '
                       'randomness in the prediction. Use sample_size=1 instead.')

    nn_model = get_trained_model()

    if args.validation_only:
        validation = load_questions_set(nn_model.token_to_index)
        validation_set_name = 'context free questions'
    else:
        eval_datasets = load_datasets(nn_model.token_to_index, nn_model.condition_to_index)
        validation = eval_datasets.cf_validation
        cs_test = eval_datasets.cs_test
        cs_test_one_condition = eval_datasets.cs_test_one_condition

        validation_set_name = 'validation set without conditions'

        _logger.info('Evaluating distinctness for context sensitive testset without conditions')
        log_distinct_metrics(nn_model, cs_test.x, samples_num=args.sample_size)

        _logger.info('Evaluating distinctness for context sensitive testset with conditions')
        log_distinct_metrics(nn_model, cs_test.x, cs_test.condition_ids, samples_num=args.sample_size)

        _logger.info('Evaluating distinctness for defined-conditions-subset without conditions')
        log_distinct_metrics(nn_model, cs_test_one_condition.x, samples_num=args.sample_size)

        _logger.info('Evaluating distinctness for defined-conditions-subset with conditions')
        log_distinct_metrics(
            nn_model, cs_test_one_condition.x, cs_test_one_condition.condition_ids, samples_num=args.sample_size)

    _logger.info('Evaluating distinctness for {}'.format(validation_set_name))
    log_distinct_metrics(nn_model, validation.x, samples_num=args.sample_size)


if __name__ == '__main__':
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        '-s', '--sample_size', action='store', default=1, type=int, help='Number of samples to average over')

    argparser.add_argument(
        '-v',
        '--validation_only',
        action='store_true',
        help='Evaluate on the validation set only (useful if you are impatient)')

    args = argparser.parse_args()

    evaluate_distinctness(args)
