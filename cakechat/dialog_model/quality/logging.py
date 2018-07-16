import os
import subprocess
import sys
import time
from collections import namedtuple
from datetime import datetime

from six.moves import xrange
import pandas as pd

# UnicodeCSV requires files to be opened as binary on Python3 by design.
# https://github.com/jdunck/python-unicodecsv/issues/65
if sys.version_info[0] == 2:
    import unicodecsv as csv
else:
    import csv

from cakechat.config import DATA_DIR, PREDICTION_MODE_FOR_TESTS, MAX_PREDICTIONS_LENGTH
from cakechat.dialog_model.inference import get_nn_responses
from cakechat.dialog_model.model_utils import transform_context_token_ids_to_sentences
from cakechat.dialog_model.quality import calculate_model_mean_perplexity, calculate_response_ngram_distinctness
from cakechat.utils.files_utils import ensure_dir
from cakechat.utils.logger import get_logger
from cakechat.utils.plotters import TensorboardMetricsPlotter

_StatsInfo = namedtuple('StatsInfo', 'start_time, iteration_num, sents_batches_num')

_TENSORBOARD_LOG_DIR = os.path.join(DATA_DIR, 'tensorboard')
_METRIC_NAMES = ['perplexity', 'unigram_distinctness', 'bigram_distinctness']

_logger = get_logger(__name__)
_tensorboard_metrics_plotter = TensorboardMetricsPlotter(_TENSORBOARD_LOG_DIR)


def _get_git_revision_short_hash():
    return subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD'])


def _get_formatted_time(seconds):
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    formatted_time = '%d:%02d:%02d' % (h, m, s)

    return formatted_time


def _get_iteration_stats(stats_info):
    stats_str = 'Batch iteration number %s\n' % str(stats_info.iteration_num)

    total_elapsed_time = time.time() - stats_info.start_time
    stats_str += 'Total elapsed time: %s\n' % _get_formatted_time(total_elapsed_time)

    elapsed_time_per_iteration = total_elapsed_time / (stats_info.iteration_num + 1)
    stats_str += 'Elapsed time for a batch: %s\n' % _get_formatted_time(elapsed_time_per_iteration)

    estimated_time_for_full_pass = elapsed_time_per_iteration * stats_info.sents_batches_num
    stats_str += 'Estimated time for a full dataset pass: %s\n' % _get_formatted_time(estimated_time_for_full_pass)

    return stats_str


def init_csv_writer(fh, output_seq_len, model_params_str):
    csv_writer = csv.writer(fh, delimiter='\t')
    csv_writer.writerow([''])  # empty row for better readability
    csv_writer.writerow([model_params_str])
    csv_writer.writerow(['commit hash: %s' % _get_git_revision_short_hash()])
    csv_writer.writerow(['date: %s' % datetime.now().strftime("%Y-%m-%d_l%H:%M")])
    csv_writer.writerow(['%d maximum tokens in the response' % output_seq_len])

    return csv_writer


def save_metrics(metrics, model_name):
    for metric_name, metric_value in metrics.items():
        _tensorboard_metrics_plotter.plot(model_name, metric_name, float(metric_value))


def calculate_and_log_val_metrics(nn_model,
                                  context_sensitive_val_subset,
                                  context_free_val,
                                  prediction_mode=PREDICTION_MODE_FOR_TESTS):
    val_metrics = dict()

    val_metrics['context_free_perplexity'] = calculate_model_mean_perplexity(nn_model, context_free_val)
    _logger.info('Current val context-free perplexity: {0:.2f}'.format(val_metrics['context_free_perplexity']))

    val_metrics['context_sensitive_perplexity'] = \
        calculate_model_mean_perplexity(nn_model, context_sensitive_val_subset)
    _logger.info('Current val context-sensitive perplexity: {0:.2f}'
                 .format(val_metrics['context_sensitive_perplexity']))

    val_metrics['unigram_distinctness'] = calculate_response_ngram_distinctness(
        context_sensitive_val_subset.x,
        nn_model,
        ngram_len=1,
        mode=prediction_mode,
        condition_ids=context_sensitive_val_subset.condition_ids)

    val_metrics['bigram_distinctness'] = calculate_response_ngram_distinctness(
        context_sensitive_val_subset.x,
        nn_model,
        ngram_len=2,
        mode=prediction_mode,
        condition_ids=context_sensitive_val_subset.condition_ids)

    _logger.info('Current val distinctness: uni={0:.3f}, bi={1:.3f}'.format(val_metrics['unigram_distinctness'],
                                                                            val_metrics['bigram_distinctness']))

    return val_metrics


def log_predictions(predictions_path,
                    x_test,
                    nn_model,
                    prediction_modes=(PREDICTION_MODE_FOR_TESTS,),
                    stats_info=None,
                    cur_perplexity=None,
                    output_seq_len=MAX_PREDICTIONS_LENGTH,
                    **kwargs):
    """
    Generate responses for provided contexts and save the results on the disk. For a given context
    several responses will be generated - one for each mode from the prediction_modes list.

    :param predictions_path: Generated responses will be saved to this file
    :param x_test: context token ids, numpy array of shape (batch_size, context_len, INPUT_SEQUENCE_LENGTH)
    :param nn_model: instance of CakeChatModel class
    :param prediction_modes: Iterable of modes to be used for responses generation. See PREDICTION_MODES
                             for available options
    :param stats_info: Info about current training status: total time passed, time spent on training,
                       processed batches number and estimated time for one epoch
    :param cur_perplexity: Addition to stats_info - current perplexity metric calculated on validation dataset
    :param output_seq_len: Max number of tokens in generated responses
    """

    _logger.info('Logging responses to test lines')

    # Create all the directories for the prediction path in case they don't exist
    prediction_dir = os.path.dirname(predictions_path)
    if prediction_dir:
        ensure_dir(prediction_dir)

    with open(predictions_path, 'w') as test_res_fh:
        csv_writer = init_csv_writer(test_res_fh, output_seq_len, nn_model.model_name)

        if cur_perplexity:
            csv_writer.writerow(['Current perplexity: %.2f' % cur_perplexity])
        if stats_info:
            csv_writer.writerow([_get_iteration_stats(stats_info)])

    contexts = transform_context_token_ids_to_sentences(x_test, nn_model.index_to_token)
    predictions_data = pd.DataFrame()
    predictions_data['contexts'] = contexts

    for pred_mode in prediction_modes:
        responses_batch = get_nn_responses(x_test, nn_model, pred_mode, **kwargs)
        # list of lists of strings, shape (contexts_num, 1)
        first_responses_batch = [response[0] for response in responses_batch]
        # list of strings, shape (contexts_num)
        predictions_data[pred_mode] = first_responses_batch

    predictions_data.to_csv(predictions_path, sep='\t', index=False, encoding='utf-8', mode='a', float_format='%.2f')

    message = '\nSuccesfully dumped {} responses.'.format(len(contexts))
    message += '\nHere they are:\n{}\n'.format(predictions_path)
    _logger.info(message)


def save_test_results(x_test, nn_model, start_time, current_batch_idx, all_batches_num, suffix='', cur_perplexity=None):

    stats_info = _StatsInfo(start_time, current_batch_idx, all_batches_num)
    results_file_name = '{}_{}.tsv'.format(nn_model.model_name, suffix)
    test_results_path = os.path.join(DATA_DIR, 'results', results_file_name)

    log_predictions(test_results_path, x_test, nn_model, stats_info=stats_info, cur_perplexity=cur_perplexity)
