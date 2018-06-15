import os
import time

from six.moves import xrange

from cakechat.config import MAX_PREDICTIONS_LENGTH, BATCH_SIZE, EPOCHES_NUM, LOG_FREQUENCY_PER_BATCHES, \
    SCREEN_LOG_FREQUENCY_PER_BATCHES, SCREEN_LOG_NUM_TEST_LINES, SHUFFLE_TRAINING_BATCHES, PREDICTION_MODE_FOR_TESTS, \
    PREDICTION_MODES, LOG_CANDIDATES_NUM, VAL_SUBSET_SIZE, LOG_LOSS_DECAY
from cakechat.dialog_model.inference import get_nn_responses
from cakechat.dialog_model.inference.service_tokens import ServiceTokensIDs
from cakechat.dialog_model.model_utils import transform_context_token_ids_to_sentences, get_training_batch, \
    get_model_full_path, reverse_nn_input
from cakechat.dialog_model.quality import save_metrics, save_test_results, calculate_and_log_val_metrics, \
    calculate_model_mean_perplexity
from cakechat.utils.dataset_loader import load_context_free_val, load_conditioned_train_set, \
    generate_subset, load_context_sensitive_val
from cakechat.utils.logger import get_logger, laconic_logger

_logger = get_logger(__name__)


def _save_model(nn_model, model_path):
    _logger.info('Saving model...')
    nn_model.save_weights(model_path)
    _logger.info('Model is saved:\n%s' % model_path)


def _delete_model(model_path):
    if os.path.isfile(model_path):
        _logger.info('Deleting old model...')
        os.remove(model_path)
        _logger.info('Model is deleted:\n%s' % model_path)


def _update_saved_nn_model(nn_model, cur_perplexities, best_perplexities, is_reverse_model=False):
    model_path = get_model_full_path(is_reverse_model)
    if all((cur < best) for cur, best in zip(cur_perplexities, best_perplexities)):
        old_suffix = '_pp_free{0:.2f}_sensitive{1:.2f}'.format(*best_perplexities)
        new_suffix = '_pp_free{0:.2f}_sensitive{1:.2f}'.format(*cur_perplexities)
        best_perplexities = cur_perplexities
        _save_model(nn_model, model_path + new_suffix)

        if new_suffix != old_suffix:
            _delete_model(model_path + old_suffix)
    else:
        _save_model(nn_model, model_path)

    return best_perplexities


def _calc_and_save_train_metrics(nn_model, train_subset, avg_loss):
    train_metrics = dict()
    train_metrics['train_perplexity'] = calculate_model_mean_perplexity(nn_model, train_subset)
    train_metrics['train_loss'] = avg_loss
    save_metrics(train_metrics)

    _logger.info('Current train perplexity: %s' % train_metrics['train_perplexity'])
    return train_metrics


def _calc_and_save_val_metrics(nn_model,
                               context_sensitive_val_subset,
                               context_free_val,
                               prediction_mode=PREDICTION_MODE_FOR_TESTS):
    val_metrics = calculate_and_log_val_metrics(nn_model, context_sensitive_val_subset, context_free_val,
                                                prediction_mode)
    save_metrics(val_metrics)

    return val_metrics


def _save_val_results(nn_model,
                      x_context_free_val,
                      x_context_sensitive_val_subset,
                      val_metrics,
                      train_info,
                      suffix='',
                      prediction_mode=PREDICTION_MODE_FOR_TESTS):
    start_time, batch_id, batches_num = train_info

    context_free_perplexity = val_metrics['context_free_perplexity'] if val_metrics else None
    context_sensitive_perplexity = val_metrics['context_sensitive_perplexity'] if val_metrics else None

    save_test_results(
        x_context_free_val,
        nn_model,
        start_time,
        batch_id,
        batches_num,
        suffix='_context_free' + suffix,
        cur_perplexity=context_free_perplexity,
        prediction_mode=prediction_mode)

    save_test_results(
        x_context_sensitive_val_subset,
        nn_model,
        start_time,
        batch_id,
        batches_num,
        suffix='_context_sensitive' + suffix,
        cur_perplexity=context_sensitive_perplexity,
        prediction_mode=prediction_mode)


def _log_sample_answers(x_test, nn_model, mode, is_reverse_model):
    _logger.info('Model: {}'.format(get_model_full_path(is_reverse_model)))
    _logger.info('Start predicting responses of length {out_len} for {n_samples} samples with mode {mode}'.format(
        out_len=MAX_PREDICTIONS_LENGTH, n_samples=x_test.shape[0], mode=mode))

    questions = transform_context_token_ids_to_sentences(x_test, nn_model.index_to_token)
    responses = get_nn_responses(x_test, nn_model, mode, output_candidates_num=LOG_CANDIDATES_NUM)
    _logger.info('Finished predicting! Logging...')

    for i, (question_ids, question) in enumerate(zip(x_test, questions)):
        laconic_logger.info('')  # for better readability
        for j, response in enumerate(responses[i]):
            laconic_logger.info('%-35s\t --#=%02d--> \t%s' % (question, j + 1, response))


def train_model(nn_model, is_reverse_model=False):
    """
    Main function fo training. Refactoring anticipated.
    """
    validation_prediction_mode = PREDICTION_MODES.sampling if is_reverse_model else PREDICTION_MODE_FOR_TESTS

    train = load_conditioned_train_set(nn_model.token_to_index, nn_model.condition_to_index)

    context_free_val = load_context_free_val(nn_model.token_to_index)

    context_sensitive_val = load_context_sensitive_val(nn_model.token_to_index, nn_model.condition_to_index)
    if is_reverse_model:
        service_tokens = ServiceTokensIDs(nn_model.token_to_index)
        train = reverse_nn_input(train, service_tokens)
        context_free_val = reverse_nn_input(context_free_val, service_tokens)
        context_sensitive_val = reverse_nn_input(context_sensitive_val, service_tokens)

    # Train subset of same size as a context-free val for metrics calculation
    train_subset = generate_subset(train, VAL_SUBSET_SIZE)

    # Context-sensitive val subset of same size as a context-free val for metrics calculation
    context_sensitive_val_subset = generate_subset(context_sensitive_val, VAL_SUBSET_SIZE)

    _logger.info('Finished preprocessing! Start training')

    batch_id = 0
    avg_loss = 0
    total_training_time = 0
    best_val_perplexities = (float('inf'), float('inf'))
    batches_num = (train.x.shape[0] - 1) / BATCH_SIZE + 1
    start_time = time.time()
    cur_val_metrics = None

    try:
        for epoches_counter in xrange(1, EPOCHES_NUM + 1):
            _logger.info('Starting epoch #%d; time = %0.2f s(training of it = %0.2f s)' %
                         (epoches_counter, time.time() - start_time, total_training_time))

            for train_batch in get_training_batch(
                [train.x, train.y, train.condition_ids], BATCH_SIZE, random_permute=SHUFFLE_TRAINING_BATCHES):
                x_train_batch, y_train_batch, condition_ids_train_batch = train_batch

                batch_id += 1
                prev_time = time.time()
                loss = nn_model.train(x_train_batch, y_train_batch, condition_ids_train_batch)

                cur_time = time.time()
                total_training_time += cur_time - prev_time
                total_time = cur_time - start_time
                avg_loss = LOG_LOSS_DECAY * avg_loss + (1 - LOG_LOSS_DECAY) * loss if batch_id > 1 else loss

                progress = 100 * float(batch_id) / batches_num
                avr_time_per_sample = total_time / batch_id
                expected_time_per_epoch = avr_time_per_sample * batches_num

                # use print here for better readability
                _logger.info('batch %s / %s (%d%%) \t'
                             'loss: %.2f \t '
                             'time: epoch %.1f h | '
                             'total %0.1f h | '
                             'train %0.1f h (%.1f%%)' %
                             (batch_id, batches_num, progress, avg_loss, expected_time_per_epoch / 3600,
                              total_time / 3600, total_training_time / 3600, 100 * total_training_time / total_time))

                if batch_id % SCREEN_LOG_FREQUENCY_PER_BATCHES == 0:
                    _log_sample_answers(context_free_val.x[:SCREEN_LOG_NUM_TEST_LINES], nn_model,
                                        validation_prediction_mode, is_reverse_model)

                if batch_id % LOG_FREQUENCY_PER_BATCHES == 0:
                    _calc_and_save_train_metrics(nn_model, train_subset, avg_loss)

                    val_metrics = _calc_and_save_val_metrics(
                        nn_model,
                        context_sensitive_val_subset,
                        context_free_val,
                        prediction_mode=validation_prediction_mode)
                    _save_val_results(
                        nn_model,
                        context_free_val.x,
                        context_sensitive_val_subset.x,
                        val_metrics,
                        train_info=(start_time, batch_id, batches_num),
                        prediction_mode=validation_prediction_mode)
                    cur_val_metrics = val_metrics

                    best_val_perplexities = \
                        _update_saved_nn_model(nn_model,
                                               (val_metrics['context_free_perplexity'],
                                                val_metrics['context_sensitive_perplexity']),
                                               best_val_perplexities,
                                               is_reverse_model=is_reverse_model)

    except (KeyboardInterrupt, SystemExit):
        _logger.info('Training cycle is stopped manually')
        _save_model(nn_model, get_model_full_path(is_reverse_model) + '_final')
        _save_val_results(
            nn_model,
            context_free_val.x,
            context_sensitive_val_subset.x,
            cur_val_metrics,
            train_info=(start_time, batch_id, batches_num),
            suffix='_final',
            prediction_mode=validation_prediction_mode)
