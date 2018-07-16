import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
from six import iteritems

from cakechat.utils.env import init_theano_env

init_theano_env()

from cakechat.utils.dataset_loader import load_datasets
from cakechat.utils.data_types import Dataset
from cakechat.utils.logger import get_tools_logger
from cakechat.dialog_model.factory import get_trained_model
from cakechat.dialog_model.model_utils import transform_token_ids_to_sentences
from cakechat.dialog_model.inference import get_nn_responses
from cakechat.dialog_model.quality import calculate_model_mean_perplexity, get_tfidf_vectorizer, \
    calculate_lexical_similarity
from cakechat.config import PREDICTION_MODE_FOR_TESTS, DEFAULT_CONDITION, RANDOM_SEED

np.random.seed(seed=RANDOM_SEED)

_logger = get_tools_logger(__file__)


def _make_non_conditioned(dataset):
    return Dataset(x=dataset.x, y=dataset.y, condition_ids=None)


def _slice_condition_data(dataset, condition_id):
    condition_mask = (dataset.condition_ids == condition_id)
    return Dataset(
        x=dataset.x[condition_mask], y=dataset.y[condition_mask], condition_ids=dataset.condition_ids[condition_mask])


def calc_perplexity_metrics(nn_model, train_subset, subset_with_conditions, validation):
    ppl_non_conditioned_train_subset = calculate_model_mean_perplexity(nn_model, _make_non_conditioned(train_subset))
    ppl_train_subset = calculate_model_mean_perplexity(nn_model, train_subset)

    ppl_non_conditioned_subset_with_conditions = calculate_model_mean_perplexity(
        nn_model, _make_non_conditioned(subset_with_conditions))
    ppl_subset_with_conditions = calculate_model_mean_perplexity(nn_model, subset_with_conditions)

    ppl_validation = calculate_model_mean_perplexity(nn_model, validation)

    return {
        'perplexity_train_subset_no_cond': ppl_non_conditioned_train_subset,
        'perplexity_train_subset': ppl_train_subset,
        'perplexity_subset_with_conditions_no_cond': ppl_non_conditioned_subset_with_conditions,
        'perplexity_subset_with_conditions': ppl_subset_with_conditions,
        'perplexity_validation': ppl_validation,
    }


def calc_perplexity_by_condition_metrics(nn_model, train):
    for condition, condition_id in nn_model.condition_to_index.items():
        if condition == DEFAULT_CONDITION:
            continue

        dataset_with_conditions = _slice_condition_data(train, condition_id)

        if not dataset_with_conditions.x.size:
            _logger.warning('No dataset samples found with the given condition "%s", skipping metrics.' % condition)
            continue

        ppl_non_conditioned = calculate_model_mean_perplexity(nn_model, _make_non_conditioned(dataset_with_conditions))
        ppl_conditioned = calculate_model_mean_perplexity(nn_model, dataset_with_conditions)

        yield condition, (ppl_non_conditioned, ppl_conditioned)


def predict_for_condition_id(nn_model, x_val, condition_id=None):
    responses = get_nn_responses(x_val, nn_model, mode=PREDICTION_MODE_FOR_TESTS, condition_ids=condition_id)
    return [candidates[0] for candidates in responses]


def calc_lexical_similarity_metrics(nn_model, train, questions, tfidf_vectorizer):
    responses_baseline = predict_for_condition_id(nn_model, questions.x)

    for condition, condition_id in nn_model.condition_to_index.items():
        if condition == DEFAULT_CONDITION:
            continue

        responses_token_ids_ground_truth = train.y[train.condition_ids == condition_id]
        if not responses_token_ids_ground_truth.size:
            _logger.warning('No dataset samples found with the given condition "%s", skipping metrics.' % condition)
            continue

        responses_ground_truth = transform_token_ids_to_sentences(responses_token_ids_ground_truth,
                                                                  nn_model.index_to_token)
        responses = predict_for_condition_id(nn_model, questions.x, condition_id)

        lex_sim_conditioned_vs_non_conditioned = calculate_lexical_similarity(responses, responses_baseline,
                                                                              tfidf_vectorizer)
        lex_sim_conditioned_vs_groundtruth = calculate_lexical_similarity(responses, responses_ground_truth,
                                                                          tfidf_vectorizer)

        yield condition, (lex_sim_conditioned_vs_non_conditioned, lex_sim_conditioned_vs_groundtruth)


if __name__ == '__main__':
    nn_model = get_trained_model()
    train, questions, validation, train_subset, conditioned_subset = load_datasets(nn_model.token_to_index,
                                                                                   nn_model.condition_to_index)
    tfidf_vectorizer = get_tfidf_vectorizer()

    for metric, perplexity in iteritems(calc_perplexity_metrics(nn_model, train_subset, conditioned_subset,
                                                      validation)):
        _logger.info('Metric: {}, perplexity: {}'.format(metric, perplexity))

    for condition, (ppl_non_conditioned, ppl_conditioned) in calc_perplexity_by_condition_metrics(nn_model, train):
        _logger.info('Condition: {}, non-conditioned perplexity: {}, conditioned perplexity: {}'.format(
            condition, ppl_non_conditioned, ppl_conditioned))

    for condition, (lex_sim_conditioned_vs_non_conditioned, lex_sim_conditioned_vs_groundtruth) in \
            calc_lexical_similarity_metrics(nn_model, train, questions, tfidf_vectorizer):
        _logger.info('Condition: {}, conditioned vs non-conditioned lexical similarity: {}'.format(
            condition, lex_sim_conditioned_vs_non_conditioned))
        _logger.info('Condition: {}, conditioned vs groundtruth lexical similarity: {}'.format(
            condition, lex_sim_conditioned_vs_groundtruth))
