import time

from cakechat.config import SCREEN_LOG_NUM_TEST_LINES, PREDICTION_MODE_FOR_TESTS, LOG_CANDIDATES_NUM, LOG_RUN_METADATA, \
    EVAL_STATE_PER_BATCHES
from cakechat.dialog_model.inference import get_nn_responses
from cakechat.dialog_model.keras_model import EvaluateAndSaveBestIntermediateModelCallback
from cakechat.dialog_model.model_utils import transform_context_token_ids_to_sentences
from cakechat.utils.dataset_loader import load_questions_set
from cakechat.utils.logger import laconic_logger


class CakeChatEvaluatorCallback(EvaluateAndSaveBestIntermediateModelCallback):
    def __init__(self,
                 model,
                 index_to_token,
                 batch_size,
                 batches_num_per_epoch,
                 eval_state_per_batches=EVAL_STATE_PER_BATCHES,
                 prediction_mode_for_tests=PREDICTION_MODE_FOR_TESTS,
                 log_run_metadata=LOG_RUN_METADATA,
                 screen_log_num_test_lines=SCREEN_LOG_NUM_TEST_LINES,
                 log_candidates_num=LOG_CANDIDATES_NUM):
        """
        :param model: CakeChatModel object
        :param eval_state_per_batches: run model evaluation each `eval_state_per_batches` steps
        """
        super(CakeChatEvaluatorCallback, self).__init__(model, eval_state_per_batches)

        self._index_to_token = index_to_token
        self._token_to_index = {v: k for k, v in index_to_token.items()}

        self._val_contexts_tokens_ids = load_questions_set(self._token_to_index).x[:screen_log_num_test_lines]
        self._val_contexts = \
            transform_context_token_ids_to_sentences(self._val_contexts_tokens_ids, self._index_to_token)

        self._batch_size = batch_size
        self._batches_num = batches_num_per_epoch

        self._cur_batch_id = 0
        self._cur_loss = 0

        self._batch_start_time = None
        self._total_training_time = 0

        # logging params
        self._prediction_mode_for_tests = prediction_mode_for_tests
        self._log_run_metadata = log_run_metadata
        self._log_candidates_num = log_candidates_num

    def on_batch_begin(self, batch, logs=None):
        self._batch_start_time = time.time()

    def on_batch_end(self, batch, logs=None):
        self._total_training_time += time.time() - self._batch_start_time

        if batch % self._eval_state_per_batches == 0:
            self._cur_loss = logs.get('loss')
            self._log_train_statistics()
            self._log_sample_answers()
            self._eval_and_save_current_model(batch)
            self._log_metrics({'train': {'loss': self._cur_loss}})

            if self._log_run_metadata:
                self._model.metrics_plotter.log_run_metadata(self._model.model_id, self._model.run_metadata)
                self._logger.info('Logged run_metadata to tensorboard')

        self._cur_batch_id += 1

    def _log_train_statistics(self):
        total_time = time.time() - self._training_start_time

        progress = self._cur_batch_id / self._batches_num  # may be more than 100% if epochs num is more than 1
        avr_time_per_batch = total_time / (self._cur_batch_id + 1)
        expected_time_per_epoch = avr_time_per_batch * self._batches_num
        total_training_time_in_percent = self._total_training_time / total_time

        batches_per_sec = self._cur_batch_id / total_time
        samples_per_sec = self._cur_batch_id * self._batch_size / total_time

        self._logger.info('Train statistics:\n')

        laconic_logger.info('batch:\t{batch_id} / {batches_num} ({progress:.1%})'.format(
            batch_id=self._cur_batch_id, batches_num=self._batches_num, progress=progress))

        laconic_logger.info('loss:\t{loss:.2f}'.format(loss=self._cur_loss))

        laconic_logger.info('time:\tepoch estimate {epoch_time:.2f} h | total {total_time:.2f} h | '
                            'train {training_time:.2f} h ({training_time_percent:.1%})'.format(
                                epoch_time=expected_time_per_epoch / 3600,  # in hours
                                total_time=total_time / 3600,  # in hours
                                training_time=self._total_training_time / 3600,  # in hours
                                training_time_percent=total_training_time_in_percent))

        laconic_logger.info('speed:\t{batches_per_hour:.0f} batches/h, {samples_per_sec:.0f} samples/sec\n'.format(
            batches_per_hour=batches_per_sec * 3600, samples_per_sec=samples_per_sec))

    def _log_sample_answers(self):
        self._logger.info('Sample responses for {} mode:'.format(self._prediction_mode_for_tests))

        responses = get_nn_responses(
            self._val_contexts_tokens_ids,
            self._model,
            self._prediction_mode_for_tests,
            output_candidates_num=self._log_candidates_num)

        for context, response_candidates in zip(self._val_contexts, responses):
            laconic_logger.info('')  # for better readability

            for i, response in enumerate(response_candidates):
                laconic_logger.info('{0: <35}\t #{1: <2d} --> \t{2}'.format(context, i + 1, response))

        laconic_logger.info('')  # for better readability
