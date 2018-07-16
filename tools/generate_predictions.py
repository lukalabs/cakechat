import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cakechat.utils.env import init_theano_env

init_theano_env()

from cakechat.dialog_model.model import get_nn_model
from cakechat.dialog_model.model_utils import transform_contexts_to_token_ids, lines_to_context
from cakechat.dialog_model.quality import log_predictions, calculate_and_log_val_metrics
from cakechat.utils.files_utils import is_non_empty_file
from cakechat.utils.logger import get_tools_logger
from cakechat.utils.dataset_loader import get_tokenized_test_lines, load_context_free_val, \
    load_context_sensitive_val
from cakechat.utils.text_processing import get_index_to_token_path, load_index_to_item, get_index_to_condition_path
from cakechat.config import BASE_CORPUS_NAME, QUESTIONS_CORPUS_NAME, INPUT_SEQUENCE_LENGTH, INPUT_CONTEXT_SIZE, \
    PREDICTION_MODES, DATA_DIR, DEFAULT_TEMPERATURE, PREDICTION_MODE_FOR_TESTS

_logger = get_tools_logger(__file__)


def _save_test_results(test_dataset, predictions_filename, nn_model, prediction_modes, **kwargs):
    test_dataset_ids = transform_contexts_to_token_ids(
        list(lines_to_context(test_dataset)), nn_model.token_to_index, INPUT_SEQUENCE_LENGTH, INPUT_CONTEXT_SIZE)

    calculate_and_log_val_metrics(nn_model,
                                  load_context_sensitive_val(nn_model.token_to_index, nn_model.condition_to_index),
                                  load_context_free_val(nn_model.token_to_index))

    log_predictions(
        predictions_filename,
        test_dataset_ids,
        nn_model,
        prediction_modes,
        **kwargs)


def predict(model_path,
            tokens_index_path=None,
            conditions_index_path=None,
            default_predictions_path=None,
            reverse_model_weights=None,
            temperatures=None,
            prediction_mode=None):

    if not tokens_index_path:
        tokens_index_path = get_index_to_token_path(BASE_CORPUS_NAME)
    if not conditions_index_path:
        conditions_index_path = get_index_to_condition_path(BASE_CORPUS_NAME)
    if not temperatures:
        temperatures = [DEFAULT_TEMPERATURE]
    if not prediction_mode:
        prediction_mode = PREDICTION_MODE_FOR_TESTS

    # Construct list of parameters values for all possible combinations of passed parameters
    prediction_params = [dict()]
    if reverse_model_weights:
        prediction_params = [
            dict(params, mmi_reverse_model_score_weight=w)
            for params in prediction_params
            for w in reverse_model_weights
        ]
    if temperatures:
        prediction_params = [dict(params, temperature=t) for params in prediction_params for t in temperatures]

    if not is_non_empty_file(tokens_index_path):
        _logger.warning('Couldn\'t find tokens_index file:\n{}. \nExiting...'.format(tokens_index_path))
        return

    index_to_token = load_index_to_item(tokens_index_path)
    index_to_condition = load_index_to_item(conditions_index_path)

    nn_model, _ = get_nn_model(index_to_token, index_to_condition, model_init_path=model_path)

    if not default_predictions_path:
        default_predictions_path = os.path.join(DATA_DIR, 'results', 'predictions_' + nn_model.model_name)

    # Get path for each combination of parameters
    predictions_paths = []
    # Add suffix to the filename only for parameters that have a specific value passed as an argument
    # If no parameters were specified, no suffix is added
    if len(prediction_params) > 1:
        for cur_params in prediction_params:
            cur_path = '{base_path}_{params_str}.tsv'.format(
                base_path=default_predictions_path,
                params_str='_'.join(['{}_{}'.format(k, v) for k, v in cur_params.items()]))
            predictions_paths.append(cur_path)
    else:
        predictions_paths = [default_predictions_path + '.tsv']

    _logger.info('Model for prediction:\n{}'.format(nn_model.model_load_path))
    _logger.info('Tokens index:\n{}'.format(tokens_index_path))
    _logger.info('File with questions:\n{}'.format(QUESTIONS_CORPUS_NAME))
    _logger.info('Files to dump responses:\n{}'.format('\n'.join(predictions_paths)))
    _logger.info('Prediction parameters\n{}'.format('\n'.join([str(x) for x in prediction_params])))

    processed_test_set = get_tokenized_test_lines(QUESTIONS_CORPUS_NAME, set(index_to_token.values()))
    processed_test_set = list(processed_test_set)

    for cur_params, cur_path in zip(prediction_params, predictions_paths):
        _logger.info('Predicting with the following params: {}'.format(cur_params))
        _save_test_results(processed_test_set, cur_path, nn_model, prediction_modes=[prediction_mode])


def parse_args():
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        '-p',
        '--prediction-mode',
        action='store',
        help='Prediction mode',
        choices=PREDICTION_MODES,
        default=None)

    argparser.add_argument(
        '-m',
        '--model',
        action='store',
        help='Path to the file with your model. '
        'Be careful, model parameters are inferred from config, not from the filename',
        default=None)

    argparser.add_argument(
        '-i',
        '--tokens_index',
        action='store',
        help='Path to the json file with index_to_token dictionary.',
        default=None)

    argparser.add_argument(
        '-c',
        '--conditions_index',
        action='store',
        help='Path to the json file with index_to_condition dictionary.',
        default=None)

    argparser.add_argument(
        '-o',
        '--output',
        action='store',
        help='Path to the file to dump predictions.'
        'Be careful, file extension ".tsv" is appended to the filename automatically',
        default=None)

    argparser.add_argument(
        '-r',
        '--reverse-model-weights',
        action='append',
        type=float,
        help='Reverse model score weight for prediction with MMI-reranking objective. Used only in *-reranking modes',
        default=None)

    argparser.add_argument(
        '-t',
        '--temperatures',
        action='append',
        help='temperature values',
        default=None,
        type=float)

    args = argparser.parse_args()

    # Extra params validation
    reranking_modes = [PREDICTION_MODES.beamsearch_reranking, PREDICTION_MODES.sampling_reranking]
    if args.reverse_model_weights and args.prediction_mode not in reranking_modes:
        raise Exception('--reverse-model-weights param can be specified only for *-reranking prediction modes.')

    return args


if __name__ == '__main__':
    args = vars(parse_args())
    predict(args.pop('model'), args.pop('tokens_index'), args.pop('conditions_index'), args.pop('output'), **args)
