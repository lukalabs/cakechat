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

    :param context_token_ids: np.array; shape (batch_size, context_size, context_len); dtype=int
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
    :return: np.array; shape (batch_size, output_candidates_num, output_seq_len); dtype=int
        Generated predictions.
    """
    if mode == PREDICTION_MODES.sampling:
        kwargs['samples_num'] = output_candidates_num

    prediction_config = PredictionConfig(mode, **kwargs)
    _logger.debug('Generating predicted response for the following params: {}'.format(prediction_config))

    predictor = predictor_factory(nn_model, mode, prediction_config.get_options_dict())
    responses = predictor.predict_responses(context_token_ids, output_seq_len, condition_ids, output_candidates_num)

    return responses


def get_nn_responses(context_token_ids,
                     nn_model,
                     mode,
                     output_candidates_num=1,
                     output_seq_len=MAX_PREDICTIONS_LENGTH,
                     condition_ids=None,
                     **kwargs):
    """
    Predicts output_candidates_num responses for every context and returns them in form of strings.
    See get_nn_response_ids for the details.

    :param context_token_ids: numpy array of integers, shape (contexts_num, INPUT_CONTEXT_SIZE, INPUT_SEQUENCE_LENGTH)
    :param nn_model: trained model
    :param mode: prediction mode, see const PREDICTION_MODES
    :param output_candidates_num: number of responses to be generated for each context
    :param output_seq_len: max length of generated responses
    :param condition_ids: extra info to be taken into account while generating response (emotion, for example)

    :return: list of lists of strings, shape (contexts_num, output_candidates_num)
    """

    response_tokens_ids = get_nn_response_ids(context_token_ids, nn_model, mode, output_candidates_num, output_seq_len,
                                              condition_ids, **kwargs)
    # shape (contexts_num, output_candidates_num, output_seq_len), numpy array of integers

    responses = [
        transform_token_ids_to_sentences(response_candidates_tokens_ids, nn_model.index_to_token)
        for response_candidates_tokens_ids in response_tokens_ids
    ]
    # responses shape (contexts_num, output_candidates_num), list of lists of strings

    return responses
