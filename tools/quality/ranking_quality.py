from __future__ import print_function

import os
import sys

from six import iteritems
from six.moves import xrange

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cakechat.utils.env import init_theano_env

init_theano_env()

from collections import defaultdict

from cakechat.config import INPUT_SEQUENCE_LENGTH, INPUT_CONTEXT_SIZE, OUTPUT_SEQUENCE_LENGTH, TEST_CORPUS_NAME, \
    TEST_DATA_DIR
from cakechat.dialog_model.inference.utils import get_sequence_score
from cakechat.dialog_model.quality import compute_retrieval_metric_mean, compute_average_precision, compute_recall_k
from cakechat.dialog_model.model_utils import transform_lines_to_token_ids, transform_contexts_to_token_ids
from cakechat.dialog_model.factory import get_trained_model
from cakechat.utils.text_processing import get_tokens_sequence
from cakechat.utils.files_utils import load_file
from cakechat.utils.data_structures import flatten


def _read_testset():
    corpus_path = os.path.join(TEST_DATA_DIR, '{}.txt'.format(TEST_CORPUS_NAME))
    test_lines = load_file(corpus_path)

    testset = defaultdict(set)
    for i in xrange(0, len(test_lines) - 1, 2):
        context = test_lines[i].strip()
        response = test_lines[i + 1].strip()
        testset[context].add(response)

    return testset


def _get_context_to_weighted_responses(nn_model, testset, all_utterances):
    token_to_index = nn_model.token_to_index

    all_utterances_ids = transform_lines_to_token_ids(
        list(map(get_tokens_sequence, all_utterances)), token_to_index, OUTPUT_SEQUENCE_LENGTH, add_start_end=True)

    context_to_weighted_responses = {}

    for context in testset:
        context_tokenized = get_tokens_sequence(context)
        repeated_context_ids = transform_contexts_to_token_ids(
            [[context_tokenized]] * len(all_utterances), token_to_index, INPUT_SEQUENCE_LENGTH, INPUT_CONTEXT_SIZE)

        scores = get_sequence_score(nn_model, repeated_context_ids, all_utterances_ids)

        context_to_weighted_responses[context] = dict(zip(all_utterances, scores))

    return context_to_weighted_responses


def _compute_metrics(model, testset):
    all_utterances = list(flatten(testset.values(), set))  # Get all unique responses
    context_to_weighted_responses = _get_context_to_weighted_responses(model, testset, all_utterances)

    test_set_size = len(all_utterances)
    metrics = {
        'mean_ap':
            compute_retrieval_metric_mean(
                compute_average_precision, testset, context_to_weighted_responses, top_count=test_set_size),
        'mean_recall@10':
            compute_retrieval_metric_mean(compute_recall_k, testset, context_to_weighted_responses, top_count=10),
        'mean_recall@25%':
            compute_retrieval_metric_mean(
                compute_recall_k, testset, context_to_weighted_responses, top_count=test_set_size // 4)
    }

    print('Test set size = %i' % test_set_size)
    for metric_name, metric_value in iteritems(metrics):
        print('%s = %s' % (metric_name, metric_value))


if __name__ == '__main__':
    nn_model = get_trained_model()
    testset = _read_testset()
    _compute_metrics(nn_model, testset)
