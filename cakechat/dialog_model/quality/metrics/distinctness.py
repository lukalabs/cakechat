import numpy as np

from cakechat.config import PREDICTION_MODE_FOR_TESTS, DEFAULT_TEMPERATURE, BEAM_SIZE, \
    PREDICTION_DISTINCTNESS_NUM_TOKENS
from cakechat.dialog_model.inference import get_nn_response_ids, ServiceTokensIDs
from cakechat.utils.profile import timer


def _calculate_distinct_ngrams(prediction_samples, ngram_len):
    """
    Takes a list of predicted token_ids and computes number of distinct ngrams
    for a given ngram_length
    """
    ngrams = set()
    for y in prediction_samples:
        # Calculate all n-grams where n = ngram_len. (Get ngram_len cyclic shifts of y and transpose the result)
        cur_ngrams = list(zip(*[y[i:] for i in range(ngram_len)]))  # yapf: disable

        # Aggregate statistics
        ngrams.update(cur_ngrams)

    return len(ngrams)


@timer
def calculate_response_ngram_distinctness(x,
                                          nn_model,
                                          ngram_len,
                                          num_tokens_to_generate=PREDICTION_DISTINCTNESS_NUM_TOKENS,
                                          mode=PREDICTION_MODE_FOR_TESTS,
                                          condition_ids=None,
                                          temperature=DEFAULT_TEMPERATURE,
                                          beam_size=BEAM_SIZE):
    """
    Computes the distinct-n metric of predictions of model given context.
     distinct-n = <number of unique n-grams> / <total number of n-grams>.

    Metric was proposed in https://arxiv.org/pdf/1510.03055v3.pdf
    """
    service_tokens_ids = ServiceTokensIDs(nn_model.token_to_index)
    num_tokens_left = num_tokens_to_generate

    responses = []
    while num_tokens_left > 0:
        # Take the first sample for each x
        responses_ids = get_nn_response_ids(
            x,
            nn_model,
            mode=mode,
            condition_ids=condition_ids,
            temperature=temperature,
            beam_size=beam_size,
            candidates_num=1)[:, 0, :]

        for y in responses_ids:
            # mask out special tokens
            mask = ~np.in1d(y, service_tokens_ids.special_tokens_ids)
            y_masked = y[mask][:num_tokens_left]
            responses.append(y_masked)

            num_tokens_left -= len(y_masked)
            if num_tokens_left == 0:
                break

    distinct_ngrams = _calculate_distinct_ngrams(responses, ngram_len)
    return distinct_ngrams / num_tokens_to_generate
