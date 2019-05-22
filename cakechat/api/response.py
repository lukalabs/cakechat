import random

from cakechat.api.config import PREDICTION_MODE, NUM_BEST_CANDIDATES_TO_PICK_FROM, SAMPLING_ATTEMPTS_NUM, \
    DEFAULT_RESPONSE
from cakechat.config import INPUT_CONTEXT_SIZE, INPUT_SEQUENCE_LENGTH, PREDICTION_MODES
from cakechat.dialog_model.factory import get_trained_model, get_reverse_model
from cakechat.dialog_model.inference import get_nn_responses, warmup_predictor
from cakechat.dialog_model.model_utils import transform_contexts_to_token_ids, transform_conditions_to_ids
from cakechat.utils.offense_detector import OffenseDetector
from cakechat.utils.offense_detector.config import OFFENSIVE_PHRASES_PATH
from cakechat.utils.text_processing import get_tokens_sequence, prettify_response

_offense_detector = OffenseDetector(OFFENSIVE_PHRASES_PATH)
_cakechat_model = get_trained_model(reverse_model=get_reverse_model(PREDICTION_MODE))
warmup_predictor(_cakechat_model, PREDICTION_MODE)


def _is_appropriate_response(response):
    return response != '' and not _offense_detector.has_offensive_ngrams(response)


def _get_non_offensive_response_using_fast_sampling(context_tokens_ids, condition_id):
    for _ in range(SAMPLING_ATTEMPTS_NUM):
        response = get_nn_responses(
            context_tokens_ids, _cakechat_model, PREDICTION_MODES.sampling, condition_ids=condition_id)[0][0]

        if _is_appropriate_response(response):
            return prettify_response(response)

    return DEFAULT_RESPONSE


def _get_non_offensive_response(context_tokens_ids, condition_id):
    responses = get_nn_responses(
        context_tokens_ids,
        _cakechat_model,
        PREDICTION_MODE,
        output_candidates_num=NUM_BEST_CANDIDATES_TO_PICK_FROM,
        condition_ids=condition_id)[0]

    responses = list(filter(_is_appropriate_response, responses))
    if responses:
        selected_response = random.choice(responses)
        return prettify_response(selected_response)

    return DEFAULT_RESPONSE


def get_response(dialog_context, emotion):
    """
    :param dialog_context: list of dialog utterances
    :param emotion: emotion to condition response
    :return: dialog response conditioned on input emotion
    """
    tokenized_dialog_context = list(map(get_tokens_sequence, dialog_context))
    tokenized_dialog_contexts = [tokenized_dialog_context]
    context_tokens_ids = transform_contexts_to_token_ids(tokenized_dialog_contexts, _cakechat_model.token_to_index,
                                                         INPUT_SEQUENCE_LENGTH, INPUT_CONTEXT_SIZE)

    condition_ids = transform_conditions_to_ids([emotion], _cakechat_model.condition_to_index, n_dialogs=1)

    if PREDICTION_MODE == PREDICTION_MODES.sampling:  # Different strategy here for better performance.
        return _get_non_offensive_response_using_fast_sampling(context_tokens_ids, condition_ids)
    else:
        return _get_non_offensive_response(context_tokens_ids, condition_ids)
