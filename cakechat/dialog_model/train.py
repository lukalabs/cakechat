import time

from six.moves import xrange

from cakechat.config import MAX_PREDICTIONS_LENGTH, BATCH_SIZE, EPOCHS_NUM, LOG_TO_TB_FREQUENCY_PER_BATCHES, \
    SCREEN_LOG_FREQUENCY_PER_BATCHES, SCREEN_LOG_NUM_TEST_LINES, SHUFFLE_TRAINING_BATCHES, PREDICTION_MODE_FOR_TESTS, \
    LOG_CANDIDATES_NUM, VAL_SUBSET_SIZE, AVG_LOSS_DECAY, LOG_TO_FILE_FREQUENCY_PER_BATCHES, \
    SAVE_MODEL_FREQUENCY_PER_BATCHES
from cakechat.dialog_model.inference import get_nn_responses
from cakechat.dialog_model.inference.service_tokens import ServiceTokensIDs
from cakechat.dialog_model.model_utils import transform_context_token_ids_to_sentences, get_training_batch, \
    reverse_nn_input
from cakechat.dialog_model.quality import save_metrics, save_test_results, calculate_and_log_val_metrics, \
    calculate_model_mean_perplexity
from cakechat.utils.data_types import TrainStats, DatasetsCollection
from cakechat.utils.dataset_loader import load_context_free_val, load_conditioned_train_set, \
    generate_subset, load_context_sensitive_val
from cakechat.utils.logger import get_logger, laconic_logger

_logger = get_logger(__name__)


def _update_saved_nn_model(nn_model, val_metrics, best_perplexities, train_stats):
    if train_stats.cur_batch_id % SAVE_MODEL_FREQUENCY_PER_BATCHES != 0:
        # don't save model on this iteration
        return best_perplexities

    # proceed with model saving
    cur_perplexities = (val_metrics['context_free_perplexity'], val_metrics['context_sensitive_perplexity'])
    is_perplexity_improved = cur_perplexities[0] < best_perplexities[0] and cur_perplexities[1] < best_perplexities[1]

    if is_perplexity_improved:
        old_suffix = '_pp_free{0:.2f}_sensitive{1:.2f}'.format(*best_perplexities)
        new_suffix = '_pp_free{0:.2f}_sensitive{1:.2f}'.format(*cur_perplexities)
        best_perplexities = cur_perplexities
        nn_model.save_model(nn_model.model_save_path + new_suffix)

        if new_suffix != old_suffix:
            nn_model.delete_model(nn_model.model_save_path + old_suffix)
    else:
        nn_model.save_model(nn_model.model_save_path)

    return best_perplexities


def _calc_and_save_train_metrics(nn_model, train_subset, avg_loss):
    train_metrics = dict()
    train_metrics['train_perplexity'] = calculate_model_mean_perplexity(nn_model, train_subset)
    train_metrics['train_loss'] = avg_loss

    save_metrics(train_metrics, nn_model.model_name)

    _logger.info('Current train perplexity: %s' % train_metrics['train_perplexity'])
    return train_metrics


def _calc_and_save_val_metrics(nn_model,
                               context_sensitive_val_subset,
                               context_free_val,
                               prediction_mode=PREDICTION_MODE_FOR_TESTS):

    val_metrics = calculate_and_log_val_metrics(nn_model, context_sensitive_val_subset, context_free_val,
                                                prediction_mode)
    save_metrics(val_metrics, nn_model.model_name)

    return val_metrics


def _save_val_results(nn_model,
                      x_context_free_val,
                      x_context_sensitive_val_subset,
                      val_metrics,
                      train_stats,
                      suffix=''):

    context_free_perplexity = val_metrics['context_free_perplexity'] if val_metrics else None
    context_sensitive_perplexity = val_metrics['context_sensitive_perplexity'] if val_metrics else None

    save_test_results(
        x_context_free_val,
        nn_model,
        train_stats.start_time,
        train_stats.cur_batch_id,
        train_stats.batches_num,
        suffix='_context_free' + suffix,
        cur_perplexity=context_free_perplexity)

    save_test_results(
        x_context_sensitive_val_subset,
        nn_model,
        train_stats.start_time,
        train_stats.cur_batch_id,
        train_stats.batches_num,
        suffix='_context_sensitive' + suffix,
        cur_perplexity=context_sensitive_perplexity)


def _log_sample_answers(x_test, nn_model, mode):
    _logger.info('Start predicting responses of length {out_len} for {n_samples} samples with mode {mode}'.format(
        out_len=MAX_PREDICTIONS_LENGTH, n_samples=x_test.shape[0], mode=mode))

    questions = transform_context_token_ids_to_sentences(x_test, nn_model.index_to_token)
    responses = get_nn_responses(x_test, nn_model, mode, output_candidates_num=LOG_CANDIDATES_NUM)
    _logger.info('Finished predicting! Logging...')

    for i, (question_ids, question) in enumerate(zip(x_test, questions)):
        laconic_logger.info('')  # for better readability
        for j, response in enumerate(responses[i]):
            laconic_logger.info('%-35s\t --#=%02d--> \t%s' % (question, j + 1, response))

    laconic_logger.info('')  # for better readability


def _log_train_info_for_one_batch(train_stats):
    total_time = (time.time() - train_stats.start_time) / 3600  # in hours
    total_train_time = train_stats.total_training_time / 3600  # in hours
    shifted_batch_id = train_stats.cur_batch_id + 1

    progress = float(shifted_batch_id) / train_stats.batches_num
    avr_time_per_batch = total_time / shifted_batch_id
    expected_time_per_epoch = avr_time_per_batch * train_stats.batches_num
    total_training_time_in_percent = total_train_time / total_time

    # use print here for better readability
    _logger.info('batch {batch_id} / {batches_num} ({progress:.1%}) \t'
                 'loss: {loss:.2f} \t '
                 'time: epoch {epoch_time:.1f} h | total {total_time:.1f} h | '
                 'train {train_time:.1f} h ({train_time_percent:.1%})'
                 .format(
                    batch_id=shifted_batch_id,
                    batches_num=train_stats.batches_num,
                    progress=progress,
                    loss=train_stats.cur_loss,
                    epoch_time=expected_time_per_epoch,
                    total_time=total_time,
                    train_time=total_train_time,
                    train_time_percent=total_training_time_in_percent
                  ))


def _analyse_model_performance_and_dump_results(nn_model, datasets, train_stats):
    cur_best_val_perplexities = train_stats.best_val_perplexities
    cur_val_metrics = train_stats.cur_val_metrics

    _log_train_info_for_one_batch(train_stats)

    if train_stats.cur_batch_id % SCREEN_LOG_FREQUENCY_PER_BATCHES == 0:
        questions_for_sampling = datasets.context_free_val.x[:SCREEN_LOG_NUM_TEST_LINES]
        _log_sample_answers(questions_for_sampling, nn_model, PREDICTION_MODE_FOR_TESTS)

    if train_stats.cur_batch_id % LOG_TO_TB_FREQUENCY_PER_BATCHES == 0:
        # save metrics on train data
        _calc_and_save_train_metrics(nn_model,
                                     datasets.train_subset,
                                     train_stats.cur_loss)

        # save metrics on validation data
        cur_val_metrics = _calc_and_save_val_metrics(
            nn_model,
            datasets.context_sensitive_val_subset,
            datasets.context_free_val,
            prediction_mode=PREDICTION_MODE_FOR_TESTS)

    if train_stats.cur_batch_id % LOG_TO_FILE_FREQUENCY_PER_BATCHES == 0:
        # save predictions on validation inputs
        _save_val_results(
            nn_model,
            datasets.context_free_val.x,
            datasets.context_sensitive_val_subset.x,
            cur_val_metrics,
            train_stats=train_stats)

    return cur_best_val_perplexities, cur_val_metrics


def _get_datasets(nn_model, is_reverse_model):
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

    datasets_collection = DatasetsCollection(
        train=train,
        train_subset=train_subset,
        context_free_val=context_free_val,
        context_sensitive_val=context_sensitive_val,
        context_sensitive_val_subset=context_sensitive_val_subset
    )

    return datasets_collection


def _get_decayed_avg_loss(avg_loss, new_loss, avg_loss_decay=AVG_LOSS_DECAY):
    return avg_loss_decay * avg_loss + (1 - avg_loss_decay) * new_loss


def train_model(nn_model):
    _logger.info('\nDefault model save path:\n{}\n'.format(nn_model.model_save_path))

    datasets_collection = _get_datasets(nn_model, nn_model.is_reverse_model)
    _logger.info('Finished preprocessing! Start training')

    batch_id = 0
    best_val_perplexities = (float('inf'), float('inf'))
    cur_val_metrics = None

    batches_num = (datasets_collection.train.x.shape[0] + BATCH_SIZE - 1) // BATCH_SIZE
    # The adding (BATCH_SIZE - 1) should be used here to count the last batch
    # that may be smaller than BATCH_SIZE

    cur_loss = 0
    total_training_time = 0
    start_time = time.time()

    for epoch_id in xrange(EPOCHS_NUM):
        _logger.info('Starting epoch #{}'.format(epoch_id))

        for train_batch in get_training_batch(datasets_collection.train,
                                              BATCH_SIZE,
                                              random_permute=SHUFFLE_TRAINING_BATCHES):
            train_stats = TrainStats(
                cur_batch_id=batch_id,
                batches_num=batches_num,
                start_time=start_time,
                total_training_time=total_training_time,
                cur_loss=cur_loss,
                best_val_perplexities=best_val_perplexities,
                cur_val_metrics=cur_val_metrics
            )

            best_val_perplexities, cur_val_metrics = \
                _analyse_model_performance_and_dump_results(nn_model, datasets_collection, train_stats)

            best_val_perplexities = \
                _update_saved_nn_model(nn_model, cur_val_metrics, best_val_perplexities, train_stats)

            prev_time = time.time()

            loss = nn_model.train(*train_batch)
            cur_loss = _get_decayed_avg_loss(cur_loss, loss) if batch_id else loss

            total_training_time += time.time() - prev_time
            batch_id += 1
