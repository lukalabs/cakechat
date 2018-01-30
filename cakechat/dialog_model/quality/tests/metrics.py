import unittest

import numpy as np

from cakechat.dialog_model.quality.metrics.perplexity import _calculate_mean_perplexity
from cakechat.dialog_model.quality.metrics.utils import MetricsException

_DELTA = 1e-3


class TestMetrics(unittest.TestCase):
    def test_perplexity_inf(self):
        output_probs = np.array([[0.2, 0.3, 0.5], [1.0, 0.0, 0.0]])
        output_tokens = np.array([0, 1])
        sequence_likelihood = np.log(output_probs[np.arange(len(output_probs)), output_tokens]).sum()

        perp = _calculate_mean_perplexity(np.array([output_tokens]), np.array([sequence_likelihood]), 2)
        perp_exp = np.inf

        self.assertAlmostEqual(perp, perp_exp, delta=_DELTA)

    def test_perplexity(self):
        output_probs = np.array([[0.2, 0.3, 0.5], [1.0, 0.0, 0.0]])
        output_tokens = np.array([0, 0])
        sequence_likelihood = np.log(output_probs[np.arange(len(output_probs)), output_tokens]).sum()

        perp = _calculate_mean_perplexity(np.array([output_tokens]), np.array([sequence_likelihood]), 2)
        perp_exp = np.exp(-0.5 * np.log(0.2))

        self.assertAlmostEqual(perp, perp_exp, delta=_DELTA)

    def test_perplexity_with_all_skips(self):
        output_probs = np.array([[0.2, 0.3, 0.5], [1.0, 0.0, 0.0]])
        output_tokens = np.array([0, 0])
        sequence_likelihood = np.log(output_probs[np.arange(len(output_probs)), output_tokens]).sum()

        skip_token_id = 0

        with self.assertRaises(MetricsException):
            _calculate_mean_perplexity(np.array([output_tokens]), np.array([sequence_likelihood]), skip_token_id)

    def test_perplexity_with_skip_token_exp(self):
        output_probs = np.array([[0.2, 0.3, 0.5], [1.0, 0.0, 0.0]])
        output_tokens = np.array([2, 1])
        skip_token_id = 1

        mask = output_tokens != skip_token_id
        sequence_likelihood = np.log(output_probs[np.arange(len(output_probs[mask])), output_tokens[mask]]).sum()

        perp_exp = np.exp(-np.log(0.5))
        perp = _calculate_mean_perplexity(np.array([output_tokens]), np.array([sequence_likelihood]), skip_token_id)

        self.assertAlmostEqual(perp, perp_exp, delta=_DELTA)


if __name__ == '__main__':
    unittest.main()
