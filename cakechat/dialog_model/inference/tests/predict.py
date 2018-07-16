import os
import sys
import unittest

import numpy as np
from six.moves import xrange

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from cakechat.utils.env import init_theano_env

init_theano_env()

from cakechat.dialog_model.factory import get_trained_model
from cakechat.dialog_model.inference import get_sequence_log_probs
from cakechat.dialog_model.inference.utils import get_next_token_log_prob_one_step
from cakechat.config import DECODER_DEPTH, HIDDEN_LAYER_DIMENSION, RANDOM_SEED

np.random.seed(seed=RANDOM_SEED)


class TestPredict(unittest.TestCase):
    @staticmethod
    def _predict_log_probabilities_one_step(nn_model, x_batch, y_batch):
        """
        Predict answers for every sequence token by token until EOS_TOKEN occurred in the sequence using sampling with temperature.
        All the rest of the sequence is filled with PAD_TOKENs.
        """
        thought_vectors_batch = nn_model.get_thought_vectors(x_batch)
        hidden_states_batch = np.zeros((x_batch.shape[0], DECODER_DEPTH, HIDDEN_LAYER_DIMENSION), dtype=np.float32)

        total_log_probs = np.zeros((y_batch.shape[0], y_batch.shape[1] - 1, nn_model.vocab_size))
        for token_idx in xrange(1, y_batch.shape[1]):
            hidden_states_batch, next_token_log_probs_batch = \
                get_next_token_log_prob_one_step(nn_model, thought_vectors_batch, hidden_states_batch,
                                                 y_batch[:, token_idx - 1], condition_ids=None)
            # total_log_probs has shape (batch_size x num_tokens x vocab_size)
            total_log_probs[:, token_idx - 1, :] = next_token_log_probs_batch

        return total_log_probs

    def test_one_step_decoder(self):
        nn_model = get_trained_model()

        _EPS = 1e-6
        batch_size = 1
        context_size = 3
        input_seq_len = 10
        output_seq_len = 9

        x = np.random.randint(0, nn_model.vocab_size, size=(batch_size, context_size, input_seq_len), dtype=np.int32)
        y = np.random.randint(0, nn_model.vocab_size, size=(batch_size, output_seq_len), dtype=np.int32)

        ground_truth_log_probabilities = get_sequence_log_probs(nn_model, x, y, condition_ids=None)
        one_step_log_probabilities = self._predict_log_probabilities_one_step(nn_model, x, y)
        mae = np.abs(one_step_log_probabilities - ground_truth_log_probabilities).mean()

        self.assertTrue(mae < _EPS)


if __name__ == '__main__':
    unittest.main()
