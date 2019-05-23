import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


from cakechat.utils.env import init_cuda_env

init_cuda_env()

from cakechat.config import QUESTIONS_CORPUS_NAME, INPUT_SEQUENCE_LENGTH, INPUT_CONTEXT_SIZE, \
    PREDICTION_MODES, PREDICTION_MODE_FOR_TESTS, DEFAULT_CONDITION, RANDOM_SEED, INTX
from cakechat.utils.text_processing import get_tokens_sequence, replace_out_of_voc_tokens
from cakechat.utils.dataset_loader import get_tokenized_test_lines
from cakechat.dialog_model.model_utils import transform_context_token_ids_to_sentences, \
    transform_contexts_to_token_ids, lines_to_context
from cakechat.dialog_model.inference import get_nn_responses
from cakechat.dialog_model.factory import get_trained_model

np.random.seed(seed=RANDOM_SEED)


def load_corpus(nn_model, corpus_name):
    return get_tokenized_test_lines(corpus_name, set(nn_model.index_to_token.values()))


def process_text(nn_model, text):
    tokenized_line = get_tokens_sequence(text)
    return [replace_out_of_voc_tokens(tokenized_line, nn_model.token_to_index)]


def transform_lines_to_contexts_token_ids(tokenized_lines, nn_model):
    return transform_contexts_to_token_ids(
        list(lines_to_context(tokenized_lines)), nn_model.token_to_index, INPUT_SEQUENCE_LENGTH, INPUT_CONTEXT_SIZE)


def predict_for_condition_id(nn_model, contexts, condition_id, prediction_mode=PREDICTION_MODE_FOR_TESTS):
    condition_ids = np.array([condition_id] * contexts.shape[0], dtype=INTX)
    responses = get_nn_responses(
        contexts, nn_model, mode=prediction_mode, output_candidates_num=1, condition_ids=condition_ids)
    return [candidates[0] for candidates in responses]


def print_predictions(nn_model, contexts_token_ids, condition, prediction_mode=PREDICTION_MODE_FOR_TESTS):
    x_sents = transform_context_token_ids_to_sentences(contexts_token_ids, nn_model.index_to_token)
    y_sents = predict_for_condition_id(
        nn_model, contexts_token_ids, nn_model.condition_to_index[condition], prediction_mode=prediction_mode)

    for x, y in zip(x_sents, y_sents):
        print('condition: {}; context: {}'.format(condition, x))
        print('response: {}'.format(y))
        print()


def parse_args():
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        '-p',
        '--prediction-mode',
        action='store',
        help='Prediction mode',
        choices=PREDICTION_MODES,
        default=PREDICTION_MODE_FOR_TESTS)

    argparser.add_argument('-d', '--data', action='store', help='Corpus name', default=QUESTIONS_CORPUS_NAME)
    argparser.add_argument('-t', '--text', action='store', help='Context message that feed to the model', default=None)
    argparser.add_argument('-c', '--condition', action='store', help='Condition', default=DEFAULT_CONDITION)

    return argparser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    nn_model = get_trained_model()

    if args.text:
        tokenized_lines = process_text(nn_model, args.text)
    else:
        tokenized_lines = load_corpus(nn_model, args.data)

    contexts_token_ids = transform_lines_to_contexts_token_ids(tokenized_lines, nn_model)

    print_predictions(nn_model, contexts_token_ids, args.condition, prediction_mode=args.prediction_mode)
