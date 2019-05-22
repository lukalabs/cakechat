import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from cakechat.utils.env import init_cuda_env

init_cuda_env()

from cakechat.utils.dataset_loader import load_datasets
from cakechat.utils.data_types import Dataset
from cakechat.utils.logger import get_tools_logger
from cakechat.dialog_model.factory import get_trained_model
from cakechat.dialog_model.model_utils import transform_token_ids_to_sentences
from cakechat.dialog_model.inference import get_nn_responses
from cakechat.dialog_model.quality import calculate_model_mean_perplexity, get_tfidf_vectorizer, \
    calculate_lexical_similarity
from cakechat.config import PREDICTION_MODE_FOR_TESTS, DEFAULT_CONDITION

_logger = get_tools_logger(__file__)


def _make_non_conditioned(dataset):
    return Dataset(x=dataset.x, y=dataset.y, condition_ids=None)


def _slice_condition_data(dataset, condition_id):
    condition_mask = (dataset.condition_ids == condition_id)
    return Dataset(
        x=dataset.x[condition_mask], y=dataset.y[condition_mask], condition_ids=dataset.condition_ids[condition_mask])


def calc_perplexity_metrics(nn_model, eval_datasets):
    return {
        'ppl_cs_test':
            calculate_model_mean_perplexity(nn_model, eval_datasets.cs_test),
        'ppl_cs_test_not_conditioned':
            calculate_model_mean_perplexity(nn_model, _make_non_conditioned(eval_datasets.cs_test)),
        'ppl_cs_test_one_condition':
            calculate_model_mean_perplexity(nn_model, eval_datasets.cs_test_one_condition),
        'ppl_cs_test_one_condition_not_conditioned':
            calculate_model_mean_perplexity(nn_model, _make_non_conditioned(eval_datasets.cs_test_one_condition)),
        'ppl_cf_validation':
            calculate_model_mean_perplexity(nn_model, eval_datasets.cf_validation)
    }


def calc_perplexity_for_conditions(nn_model, dataset):
    cond_to_ppl_conditioned, cond_to_ppl_not_conditioned = {}, {}

    for condition, condition_id in nn_model.condition_to_index.items():
        if condition == DEFAULT_CONDITION:
            continue

        dataset_with_conditions = _slice_condition_data(dataset, condition_id)

        if not dataset_with_conditions.x.size:
            _logger.warning(
                'No dataset samples found with the given condition "{}", skipping metrics.'.format(condition))
            continue

        cond_to_ppl_conditioned[condition] = \
            calculate_model_mean_perplexity(nn_model, _make_non_conditioned(dataset_with_conditions))

        cond_to_ppl_not_conditioned[condition] = \
            calculate_model_mean_perplexity(nn_model, dataset_with_conditions)

    return cond_to_ppl_conditioned, cond_to_ppl_not_conditioned


def predict_for_condition_id(nn_model, x_val, condition_id=None):
    responses = get_nn_responses(x_val, nn_model, mode=PREDICTION_MODE_FOR_TESTS, condition_ids=condition_id)
    return [candidates[0] for candidates in responses]


def calc_lexical_similarity_metrics(nn_model, testset, tfidf_vectorizer):
    """
    For each condition calculate lexical similarity between ground-truth responses and
    generated conditioned responses. Similarly compare ground-truth responses with non-conditioned generated responses.
    If lex_sim(gt, cond_resp) > lex_sim(gt, non_cond_resp), the conditioning on extra information proves to be useful.
    :param nn_model: trained model to evaluate
    :param testset: context-sensitive testset, instance of Dataset
    :param tfidf_vectorizer: instance of scikit-learn TfidfVectorizer, calculates lexical similariry for documents
    according to TF-IDF metric
    :return: two dictionaries:
        {condition: lex_sim(gt, cond_resp)},
        {condition: lex_sim(gt, non_cond_resp)}
    """
    gt_vs_cond_lex_sim, gt_vs_non_cond_lex_sim = {}, {}

    for condition, condition_id in nn_model.condition_to_index.items():
        sample_mask_for_condition = testset.condition_ids == condition_id
        contexts_for_condition = testset.x[sample_mask_for_condition]
        responses_for_condition = testset.y[sample_mask_for_condition]

        if not responses_for_condition.size:
            _logger.warning('No dataset samples found for condition "{}", skip it.'.format(condition))
            continue

        gt_responses = transform_token_ids_to_sentences(responses_for_condition, nn_model.index_to_token)
        conditioned_responses = predict_for_condition_id(nn_model, contexts_for_condition, condition_id)
        non_conditioned_responses = predict_for_condition_id(nn_model, contexts_for_condition, condition_id=None)

        gt_vs_cond_lex_sim[condition] = \
            calculate_lexical_similarity(gt_responses, conditioned_responses, tfidf_vectorizer)

        gt_vs_non_cond_lex_sim[condition] = \
            calculate_lexical_similarity(gt_responses, non_conditioned_responses, tfidf_vectorizer)

    return gt_vs_cond_lex_sim, gt_vs_non_cond_lex_sim


if __name__ == '__main__':
    nn_model = get_trained_model()
    eval_datasets = load_datasets(nn_model.token_to_index, nn_model.condition_to_index)

    print('\nPerplexity on datasets:')
    for dataset, perplexity in calc_perplexity_metrics(nn_model, eval_datasets).items():
        print('\t{}: \t{:.1f}'.format(dataset, perplexity))

    cond_to_ppl_conditioned, cond_to_ppl_not_conditioned = \
        calc_perplexity_for_conditions(nn_model, eval_datasets.cs_test)

    print('\nPerplexity on conditioned testset for conditions:')
    for condition, perplexity in cond_to_ppl_conditioned.items():
        print('\t{}: \t{:.1f}'.format(condition, perplexity))

    print('\nPerplexity on non-conditioned testset for conditions:')
    for condition, perplexity in cond_to_ppl_not_conditioned.items():
        print('\t{}: \t{:.1f}'.format(condition, perplexity))

    gt_vs_cond_lex_sim, gt_vs_non_cond_lex_sim = \
        calc_lexical_similarity_metrics(nn_model, eval_datasets.cs_test, get_tfidf_vectorizer())

    print('\nLexical similarity, ground-truth vs. conditioned responses:')
    for condition, lex_sim in gt_vs_cond_lex_sim.items():
        print('\t{}: \t{:.2f}'.format(condition, lex_sim))

    print('\nLexical similarity, ground-truth vs. non-conditioned responses:')
    for condition, lex_sim in gt_vs_non_cond_lex_sim.items():
        print('\t{}: \t{:.2f}'.format(condition, lex_sim))
