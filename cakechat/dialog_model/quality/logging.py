import os
import subprocess
import sys
import time
from collections import namedtuple
from datetime import datetime

from six.moves import xrange

# UnicodeCSV requires files to be opened as binary on Python3 by design.
# https://github.com/jdunck/python-unicodecsv/issues/65
if sys.version_info[0] == 2:
    import unicodecsv as csv
else:
    import csv

from cakechat.config import DATA_DIR, PREDICTION_MODE_FOR_TESTS, LOG_CANDIDATES_NUM, MAX_PREDICTIONS_LENGTH
from cakechat.dialog_model.inference import get_nn_responses
from cakechat.dialog_model.model_utils import transform_context_token_ids_to_sentences, get_model_full_params_str
from cakechat.dialog_model.quality import calculate_model_mean_perplexity, calculate_response_ngram_distinctness
from cakechat.utils.files_utils import ensure_dir
from cakechat.utils.logger import get_logger
from cakechat.utils.plotters import TensorboardMetricsPlotter

_StatsInfo = namedtuple('StatsInfo', 'start_time, iteration_num, sents_batches_num')

_NN_MODEL_PARAMS_STR = get_model_full_params_str()

_TENSORBOARD_LOG_DIR = os.path.join(DATA_DIR, 'tensorboard')
_TEST_RESULTS_PATH = os.path.join(DATA_DIR, 'results', _NN_MODEL_PARAMS_STR + '.tsv')
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

    elapsed_time_per_iteration = total_elapsed_time / stats_info.iteration_num
    stats_str += 'Elapsed time for a batch: %s\n' % _get_formatted_time(elapsed_time_per_iteration)

    estimated_time_for_full_pass = elapsed_time_per_iteration * stats_info.sents_batches_num
    stats_str += 'Estimated time for a full dataset pass: %s\n' % _get_formatted_time(estimated_time_for_full_pass)

    return stats_str


def init_csv_writer(fh, mode, output_seq_len):
    csv_writer = csv.writer(fh, delimiter='\t')
    csv_writer.writerow([''])  # empty row for better readability
    csv_writer.writerow([_NN_MODEL_PARAMS_STR])
    csv_writer.writerow(['commit hash: %s' % _get_git_revision_short_hash()])
    csv_writer.writerow(['date: %s' % datetime.now().strftime("%Y-%m-%d_l%H:%M")])
    csv_writer.writerow(['Prediction using %s' % mode])
    csv_writer.writerow(['%d maximum tokens in the response' % output_seq_len])

    return csv_writer


def save_metrics(metrics):
    for name, value in metrics.items():
        _tensorboard_metrics_plotter.plot(model_name=_NN_MODEL_PARAMS_STR, metric_name=name, metric_value=float(value))


def calculate_and_log_val_metrics(nn_model,
                                  context_sensitive_val_subset,
                                  context_free_val,
                                  prediction_mode=PREDICTION_MODE_FOR_TESTS):
    val_metrics = dict()
    val_metrics['context_sensitive_perplexity'] = calculate_model_mean_perplexity(nn_model,
                                                                                  context_sensitive_val_subset)
    val_metrics['context_free_perplexity'] = calculate_model_mean_perplexity(nn_model, context_free_val)

    val_metrics['unigram_distinctness'] = calculate_response_ngram_distinctness(
        context_sensitive_val_subset.x, nn_model, ngram_len=1, mode=prediction_mode)
    val_metrics['bigram_distinctness'] = calculate_response_ngram_distinctness(
        context_sensitive_val_subset.x, nn_model, ngram_len=2, mode=prediction_mode)

    _logger.info('Current val context-sensitive perplexity: %s' % val_metrics['context_sensitive_perplexity'])
    _logger.info('Current val context-free perplexity: %s' % val_metrics['context_free_perplexity'])
    _logger.info('Current val distinctness: uni=%s, bi=%s' % (val_metrics['unigram_distinctness'],
                                                              val_metrics['bigram_distinctness']))
    return val_metrics


def log_predictions(predictions_path,
                    x_test,
                    nn_model,
                    mode,
                    candidates_num=None,
                    stats_info=None,
                    cur_perplexity=None,
                    output_seq_len=MAX_PREDICTIONS_LENGTH,
                    **kwargs):
    _logger.info('Logging responses to test lines')

    # Create all the directories for the prediction path in case they don't exist
    prediction_dir = os.path.dirname(predictions_path)
    if prediction_dir:
        ensure_dir(prediction_dir)

    with open(predictions_path, 'w') as test_res_fh:
        csv_writer = init_csv_writer(test_res_fh, mode, output_seq_len)

        if cur_perplexity:
            csv_writer.writerow(['Current perplexity: %.2f' % cur_perplexity])
        if stats_info:
            csv_writer.writerow([_get_iteration_stats(stats_info)])

        csv_writer.writerow(['input sentence'] + ['candidate #{}'.format(v + 1) for v in xrange(candidates_num)])

        questions = transform_context_token_ids_to_sentences(x_test, nn_model.index_to_token)

        _logger.info('Start predicting responses of length {out_len} for {n_samples} samples with mode {mode}'.format(
            out_len=output_seq_len, n_samples=x_test.shape[0], mode=mode))

        nn_responses = get_nn_responses(x_test, nn_model, mode, candidates_num, output_seq_len, **kwargs)

        _logger.info('Logging generated predictions...')
        for idx, (question, responses) in enumerate(zip(questions, nn_responses)):
            csv_writer.writerow([question] + responses)

        _logger.info('Succesfully dumped {n_resp} of {n_resp} responses'.format(n_resp=len(questions)))
        _logger.info('Here they are: {}'.format(predictions_path))


def save_test_results(x_test,
                      nn_model,
                      start_time,
                      current_batch_idx,
                      all_batches_num,
                      suffix='',
                      cur_perplexity=None,
                      prediction_mode=PREDICTION_MODE_FOR_TESTS):
    stats_info = _StatsInfo(start_time, current_batch_idx, all_batches_num)
    test_results_path = _TEST_RESULTS_PATH.replace('.tsv', '%s.tsv' % suffix)

    log_predictions(
        test_results_path,
        x_test,
        nn_model,
        mode=prediction_mode,
        candidates_num=LOG_CANDIDATES_NUM,
        stats_info=stats_info,
        cur_perplexity=cur_perplexity)
