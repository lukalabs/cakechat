import numpy as np

from cakechat.dialog_model.inference import get_sequence_score
from cakechat.dialog_model.quality.metrics.utils import MetricsException
from cakechat.utils.logger import get_logger
from cakechat.utils.text_processing import SPECIAL_TOKENS

_logger = get_logger(__name__)


def _calculate_mean_perplexity(output_ids, output_scores, skip_token_id):
    total_nonpad_tokens = np.sum(output_ids != skip_token_id, axis=1)

    empty_sequences_mask = total_nonpad_tokens == 0
    empty_sequences_num = np.sum(empty_sequences_mask)
    non_empty_sequences_mask = ~empty_sequences_mask

    if empty_sequences_num:
        _logger.info('Got pads-only sequences while computing perplexity. '
                     'Skipping these {} samples'.format(empty_sequences_num))
    if np.all(empty_sequences_mask):
        raise MetricsException('Got all pad-only sequences while computing perplexity')

    output_scores = output_scores[non_empty_sequences_mask]
    total_nonpad_tokens = total_nonpad_tokens[non_empty_sequences_mask]
    sample_perplexities = np.exp(-output_scores / total_nonpad_tokens)

    return np.mean(sample_perplexities)


def calculate_model_mean_perplexity(nn_model, dataset):
    output_scores = get_sequence_score(nn_model, dataset.x, dataset.y, dataset.condition_ids)
    if not np.all(dataset.y[:, 0] == nn_model.token_to_index[SPECIAL_TOKENS.START_TOKEN]):
        raise MetricsException('All responses in the dataset have to start with start_token_id.'
                               'Make sure there are start_token ids in the beginning of each sequence in dataset.y')

    output_ids = dataset.y[:, 1:]  # remove start token

    return _calculate_mean_perplexity(output_ids, output_scores, nn_model.skip_token_id)
