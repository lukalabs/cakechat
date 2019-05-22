import csv
import os
from datetime import datetime

import pandas as pd

from cakechat.config import PREDICTION_MODE_FOR_TESTS, MAX_PREDICTIONS_LENGTH
from cakechat.dialog_model.inference import get_nn_responses
from cakechat.dialog_model.model_utils import transform_context_token_ids_to_sentences
from cakechat.dialog_model.quality import calculate_model_mean_perplexity, calculate_response_ngram_distinctness
from cakechat.utils.files_utils import ensure_dir
from cakechat.utils.logger import get_logger

_logger = get_logger(__name__)


def calculate_and_log_val_metrics(nn_model,
                                  context_sensitive_val,
                                  context_free_val,
                                  prediction_mode=PREDICTION_MODE_FOR_TESTS,
                                  calculate_ngram_distance=True):
    metric_name_to_value = {
        'context_free_perplexity': calculate_model_mean_perplexity(nn_model, context_free_val),
        'context_sensitive_perplexity': calculate_model_mean_perplexity(nn_model, context_sensitive_val)
    }

    if calculate_ngram_distance:
        for metric_name, ngram_len in [('unigram_distinctness', 1), ('bigram_distinctness', 2)]:
            metric_name_to_value[metric_name] = calculate_response_ngram_distinctness(
                context_sensitive_val.x,
                nn_model,
                ngram_len=ngram_len,
                mode=prediction_mode,
                condition_ids=context_sensitive_val.condition_ids)

    for metric_name, metric_value in metric_name_to_value.items():
        _logger.info('Val set {}: {:.3f}'.format(metric_name, metric_value))

    return metric_name_to_value


def _init_csv_writer(predictions_path, output_seq_len, model_name):
    with open(predictions_path, 'w', encoding='utf-8') as fh:
        csv_writer = csv.writer(fh, delimiter='\t')
        csv_writer.writerow([model_name])
        csv_writer.writerow(['date: {}'.format(datetime.now().strftime('%Y-%m-%d %H:%M'))])
        csv_writer.writerow(['{} maximum tokens in the response'.format(output_seq_len)])
        csv_writer.writerow([''])  # empty row for better readability


def log_predictions(predictions_path,
                    contexts_token_ids,
                    nn_model,
                    prediction_modes,
                    output_seq_len=MAX_PREDICTIONS_LENGTH,
                    **kwargs):
    """
    Generate responses for provided contexts and save the results on the disk. For a given context
    several responses will be generated - one for each mode from the prediction_modes list.

    :param predictions_path: Generated responses will be saved to this file
    :param contexts_token_ids: contexts token ids, numpy array of shape (batch_size, context_len, INPUT_SEQUENCE_LENGTH)
    :param nn_model: instance of CakeChatModel class
    :param prediction_modes: See PREDICTION_MODES for available options
    :param output_seq_len: Max number of tokens in generated responses
    """
    _logger.info('Logging responses for test set')

    # Create all the directories for the prediction path in case they don't exist
    ensure_dir(os.path.dirname(predictions_path))

    _init_csv_writer(predictions_path, output_seq_len, nn_model.model_name)

    contexts = transform_context_token_ids_to_sentences(contexts_token_ids, nn_model.index_to_token)
    predictions_data = pd.DataFrame()
    predictions_data['contexts'] = contexts

    for prediction_mode in prediction_modes:
        predicted_responses = get_nn_responses(contexts_token_ids, nn_model, prediction_mode, **kwargs)
        # list of lists of strings, shape (contexts_num, 1)
        predicted_responses = [response[0] for response in predicted_responses]
        # list of strings, shape (contexts_num)
        predictions_data[prediction_mode] = predicted_responses

    predictions_data.to_csv(predictions_path, sep='\t', index=False, encoding='utf-8', mode='a', float_format='%.2f')

    _logger.info('Dumped {} predicted responses to {}'.format(len(contexts), predictions_path))
