import os
import sys
import unittest

import keras.backend as K
import numpy as np

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from cakechat.utils.env import init_cuda_env

init_cuda_env()

from cakechat.dialog_model.factory import get_trained_model
from cakechat.dialog_model.inference import get_sequence_log_probs
from cakechat.dialog_model.inference.utils import get_next_token_log_prob_one_step
from cakechat.config import HIDDEN_LAYER_DIMENSION, RANDOM_SEED, INPUT_CONTEXT_SIZE, \
    INPUT_SEQUENCE_LENGTH, OUTPUT_SEQUENCE_LENGTH, INTX

np.random.seed(seed=RANDOM_SEED)


class TestPredict(unittest.TestCase):
    @staticmethod
    def _predict_log_probabilities_one_step(nn_model, x_batch, y_batch):
        """
        Predict answers for every sequence token by token until EOS_TOKEN occurred in the sequence using sampling with temperature.
        All the rest of the sequence is filled with PAD_TOKENs.
        """
        thought_vectors_batch = nn_model.get_thought_vectors(x_batch)
        hidden_states_batch = np.zeros(
            (x_batch.shape[0], nn_model.decoder_depth, HIDDEN_LAYER_DIMENSION), dtype=K.floatx())

        total_log_probs = np.zeros((y_batch.shape[0], y_batch.shape[1] - 1, nn_model.vocab_size))
        for token_idx in range(1, y_batch.shape[1]):
            hidden_states_batch, next_token_log_probs_batch = \
                get_next_token_log_prob_one_step(nn_model, thought_vectors_batch, hidden_states_batch,
                                                 y_batch[:, token_idx - 1], condition_ids=None)
            # total_log_probs has shape (batch_size x num_tokens x vocab_size)
            total_log_probs[:, token_idx - 1, :] = next_token_log_probs_batch

        return total_log_probs

    def test_one_step_decoder(self):
        nn_model = get_trained_model()

        _EPS = 1e-5
        batch_size = 1
        # input batches shapes should correspond to the shapes of the trained model layers
        context_size = INPUT_CONTEXT_SIZE
        input_seq_len = INPUT_SEQUENCE_LENGTH
        output_seq_len = OUTPUT_SEQUENCE_LENGTH

        x = np.random.randint(0, nn_model.vocab_size, size=(batch_size, context_size, input_seq_len), dtype=INTX)
        y = np.random.randint(0, nn_model.vocab_size, size=(batch_size, output_seq_len), dtype=INTX)

        ground_truth_log_probabilities = get_sequence_log_probs(nn_model, x, y, condition_ids=None)
        one_step_log_probabilities = self._predict_log_probabilities_one_step(nn_model, x, y)
        mae = np.abs(one_step_log_probabilities - ground_truth_log_probabilities).mean()

        self.assertTrue(mae < _EPS)


if __name__ == '__main__':
    unittest.main()
