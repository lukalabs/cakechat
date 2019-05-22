from cakechat.config import PREDICTION_MODES
from cakechat.dialog_model.inference.candidates import BeamsearchCandidatesGenerator, SamplingCandidatesGenerator
from cakechat.dialog_model.inference.predictor import Predictor
from cakechat.dialog_model.inference.reranking import DummyReranker, MMIReranker


def predictor_factory(nn_model, mode, config):
    """

    :param nn_model: Model used for predicting
    :param mode: Prediction mode: 'sampling', 'sampling-reranking' or 'candidates'
    :param config: All additional prediction parameters. See PredictionConfig for the details.
    :return: BasePredictor descendant with predict_response() method implemented.
    """
    if mode not in PREDICTION_MODES:
        raise ValueError('Unknown prediction mode {}. Use one of the following: {}.'.format(
            mode, list(PREDICTION_MODES)))

    if mode in [PREDICTION_MODES.beamsearch, PREDICTION_MODES.beamsearch_reranking]:
        candidates_generator = BeamsearchCandidatesGenerator(nn_model, config['beam_size'],
                                                             config['repetition_penalization_coefficient'])
    else:
        candidates_generator = SamplingCandidatesGenerator(nn_model, config['temperature'], config['samples_num'],
                                                           config['repetition_penalization_coefficient'])

    if mode in [PREDICTION_MODES.beamsearch_reranking, PREDICTION_MODES.sampling_reranking]:
        if config['mmi_reverse_model_score_weight'] <= 0:
            raise ValueError('mmi_reverse_model_score_weight should be > 0 for reranking mode')

        reranker = MMIReranker(nn_model, nn_model.reverse_model, config['mmi_reverse_model_score_weight'],
                               config['repetition_penalization_coefficient'])
    else:
        reranker = DummyReranker()

    return Predictor(nn_model, candidates_generator, reranker)
