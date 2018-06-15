import numpy as np
from six.moves import xrange

from cakechat.config import MAX_PREDICTIONS_LENGTH, BEAM_SIZE, MMI_REVERSE_MODEL_SCORE_WEIGHT, DEFAULT_TEMPERATURE, \
    SAMPLES_NUM_FOR_RERANKING, PREDICTION_MODES, REPETITION_PENALIZE_COEFFICIENT
from cakechat.dialog_model.inference.factory import predictor_factory
from cakechat.dialog_model.model_utils import transform_token_ids_to_sentences
from cakechat.utils.logger import get_logger

_logger = get_logger(__name__)


class PredictionConfig(object):
    def __init__(self, mode, **kwargs):
        self.mode = mode
        self.repetition_penalization_coefficient = \
            kwargs.get('repetition_penalization_coefficient', REPETITION_PENALIZE_COEFFICIENT)

        if self.mode == PREDICTION_MODES.sampling:
            self.temperature = kwargs.get('temperature', DEFAULT_TEMPERATURE)
            self.samples_num = kwargs.get('samples_num', 1)
        elif self.mode == PREDICTION_MODES.beamsearch:
            self.beam_size = kwargs.get('beam_size', BEAM_SIZE)
        elif self.mode == PREDICTION_MODES.sampling_reranking:
            self.temperature = kwargs.get('temperature', DEFAULT_TEMPERATURE)
            self.samples_num = kwargs.get('samples_num', SAMPLES_NUM_FOR_RERANKING)
            self.mmi_reverse_model_score_weight = kwargs.get('mmi_reverse_model_score_weight',
                                                             MMI_REVERSE_MODEL_SCORE_WEIGHT)
        elif self.mode == PREDICTION_MODES.beamsearch_reranking:
            self.beam_size = kwargs.get('beam_size', BEAM_SIZE)
            self.mmi_reverse_model_score_weight = kwargs.get('mmi_reverse_model_score_weight',
                                                             MMI_REVERSE_MODEL_SCORE_WEIGHT)

    def get_options_dict(self):
        return self.__dict__

    def __str__(self):
        return str(self.__dict__)


def warmup_predictor(nn_model, prediction_mode):
    if prediction_mode in {PREDICTION_MODES.beamsearch_reranking, PREDICTION_MODES.sampling_reranking}:
        prediction_config = PredictionConfig(prediction_mode)
        predictor_factory(nn_model, prediction_mode, prediction_config.get_options_dict())


def get_nn_response_ids(context_token_ids,
                        nn_model,
                        mode,
                        output_candidates_num=1,
                        output_seq_len=MAX_PREDICTIONS_LENGTH,
                        condition_ids=None,
                        **kwargs):
    """
    Predicts several responses for every context.

    :param context_token_ids: np.array; shape=(batch_size x context_size x context_len); dtype=int
        Represents all tokens ids to use for predicting
    :param nn_model: CakeChatModel
    :param mode: one of PREDICTION_MODES mode
    :param output_candidates_num: Number of candidates to generate.
        When mode is either 'beamsearch', 'beamsearch-reranking'  or 'sampling-reranking', the candidates with the
        highest score are returned. When mode is 'sampling', the candidates_num of samples are generated independently.
    :param condition_ids: List with ids of conditions responding for each context.
    :param output_seq_len: Number of tokens to generate.
    :param kwargs: Other prediction parameters, passed into predictor constructor.
        Might be different depending on mode. See PredictionConfig for the details.
    :return: np.array; shape=(responses_num x output_candidates_num x output_seq_len); dtype=int
        Generated predictions.
    """
    if mode == PREDICTION_MODES.sampling:
        kwargs['samples_num'] = output_candidates_num

    prediction_config = PredictionConfig(mode, **kwargs)
    _logger.debug('Generating predicted response for the following params: %s' % prediction_config)

    predictor = predictor_factory(nn_model, mode, prediction_config.get_options_dict())
    return np.array(
        predictor.predict_responses(context_token_ids, output_seq_len, condition_ids, output_candidates_num))


def get_nn_responses(context_token_ids,
                     nn_model,
                     mode,
                     output_candidates_num=1,
                     output_seq_len=MAX_PREDICTIONS_LENGTH,
                     condition_ids=None,
                     **kwargs):
    """
    Predicts several responses for every context and returns them as proccessed strings.
    See get_nn_response_ids for the details.

    :return: list of lists of strings
        Generated predictions.
    """
    response_tokens_ids = get_nn_response_ids(context_token_ids, nn_model, mode, output_candidates_num, output_seq_len,
                                              condition_ids, **kwargs)
    # Reshape to get list of lines to supply into transform_token_ids_to_sentences
    response_tokens_ids = np.reshape(response_tokens_ids, (-1, output_seq_len))
    response_tokens = transform_token_ids_to_sentences(response_tokens_ids, nn_model.index_to_token)

    lines_num = len(response_tokens) // output_candidates_num
    responses = [response_tokens[i * output_candidates_num:(i + 1) * output_candidates_num] for i in xrange(lines_num)]

    return responses
