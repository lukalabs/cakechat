import os
import sys
import unittest

import numpy as np
from scipy.stats import binom
from six.moves import xrange

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from cakechat.dialog_model.inference.candidates.sampling import TokenSampler
from cakechat.config import REPETITION_PENALIZE_COEFFICIENT, RANDOM_SEED

np.random.seed(seed=RANDOM_SEED)

# Type I error rate: probability that a test will fail even though everything is OK
# The lower the probability is the more inaccurate (in terms of Type II error) the test becomes.
# This is independent probability for every test in the TestCase.
_CONFIDENCE_LEVEL = 1e-6
# Number of samples for monte-carlo estimation or probabilities.
# The bigger number of sample is, the more accurate tests become
_SAMPLES_NUM = 10000


class TestSampling(unittest.TestCase):
    def test_sample_list(self):
        # Error rate is p(token1) + p(token2) = conf_level / 2 + conf_level / 2 = conf_level:
        probs = [_CONFIDENCE_LEVEL / 2, _CONFIDENCE_LEVEL / 2, 1 - _CONFIDENCE_LEVEL]

        token_sampler = TokenSampler(
            batch_size=1,
            banned_tokens_ids=[],
            non_penalizable_tokens_ids=range(len(probs)),
            repetition_penalization_coefficient=REPETITION_PENALIZE_COEFFICIENT)
        expected_token_ids = np.array([2])
        actual_token_ids = token_sampler.sample(probs, sample_idx=0)
        self.assertEqual(expected_token_ids, actual_token_ids)

    def test_sample_ndarray(self):
        # Error rate is p(token1) + p(token2) = conf_level / 2 + conf_level / 2 = conf_level
        probs = np.array([_CONFIDENCE_LEVEL / 2, _CONFIDENCE_LEVEL / 2, 1 - _CONFIDENCE_LEVEL], dtype=np.float32)

        token_sampler = TokenSampler(
            batch_size=1,
            banned_tokens_ids=[],
            non_penalizable_tokens_ids=range(len(probs)),
            repetition_penalization_coefficient=REPETITION_PENALIZE_COEFFICIENT)
        expected_token_ids = np.array([2])
        actual_token_ids = token_sampler.sample(probs, sample_idx=0)
        self.assertEqual(expected_token_ids, actual_token_ids)

    def test_sample_probs(self):
        probs = [0.3, 0.6, 0.1]

        token_sampler = TokenSampler(
            batch_size=1,
            banned_tokens_ids=[],
            non_penalizable_tokens_ids=range(len(probs)),
            repetition_penalization_coefficient=REPETITION_PENALIZE_COEFFICIENT)
        adjusted_confidence_level = _CONFIDENCE_LEVEL / len(probs)  # bonferroni correction
        confidence_intervals = [binom.interval(1 - adjusted_confidence_level, _SAMPLES_NUM, p) for p in probs]
        est_probs_from, est_probs_to = zip(*confidence_intervals)
        samples = np.array([token_sampler.sample(probs, 0) for _ in xrange(_SAMPLES_NUM)])
        counts = {val: np.sum(samples == val) for val in np.unique(samples)}

        for i, _ in enumerate(probs):
            self.assertLessEqual(counts[i], est_probs_to[i])
            self.assertGreaterEqual(counts[i], est_probs_from[i])

    def test_sample_with_zeros(self):
        probs = np.array([1.0, 0, 0], dtype=np.float32)

        token_sampler = TokenSampler(
            batch_size=1,
            banned_tokens_ids=[],
            non_penalizable_tokens_ids=range(len(probs)),
            repetition_penalization_coefficient=REPETITION_PENALIZE_COEFFICIENT)
        expected_token_ids = np.array([0])
        actual_token_ids = token_sampler.sample(probs, sample_idx=0)
        self.assertEqual(expected_token_ids, actual_token_ids)

    def test_sample_banned_tokens(self):
        eps = _CONFIDENCE_LEVEL * 0.3
        # Here we multiply the confidence level by 0.3 so that after removal of banned token and renormalization
        # the probability of an error remains equal to _CONFIDENCE_LEVEL value.
        probs = np.array([0.7, 0.3 - eps, eps], dtype=np.float32)

        token_sampler = TokenSampler(
            batch_size=1,
            banned_tokens_ids=[0],
            non_penalizable_tokens_ids=range(len(probs)),
            repetition_penalization_coefficient=REPETITION_PENALIZE_COEFFICIENT)
        expected_token_ids = np.array([1])
        actual_token_ids = token_sampler.sample(probs, sample_idx=0)
        self.assertEqual(expected_token_ids, actual_token_ids)

    def test_sample_banned_tokens_2(self):
        eps = 1e-6
        probs = np.array([1.0 - eps, eps, 0], dtype=np.float32)

        token_sampler = TokenSampler(
            batch_size=1,
            banned_tokens_ids=[0],
            non_penalizable_tokens_ids=range(len(probs)),
            repetition_penalization_coefficient=REPETITION_PENALIZE_COEFFICIENT)
        # Token #1 has to be returned even though its probability is really small
        expected_token_ids = np.array([1])
        actual_token_ids = token_sampler.sample(probs, sample_idx=0)
        self.assertEqual(expected_token_ids, actual_token_ids)

    def test_repetition_penalization(self):
        probs = [0.5, 0.5]

        actual_num_nonequal_pairs = 0
        for _ in xrange(_SAMPLES_NUM):
            token_sampler = TokenSampler(
                batch_size=1,
                banned_tokens_ids=[],
                non_penalizable_tokens_ids=[],
                repetition_penalization_coefficient=REPETITION_PENALIZE_COEFFICIENT)
            first_token = token_sampler.sample(probs, sample_idx=0)
            second_token = token_sampler.sample(probs, sample_idx=0)
            actual_num_nonequal_pairs += int(first_token != second_token)

        # P(first != second) = P(first=0, second=1) + P(first=1, second=0) =
        # = 0.5 * 0.5 * r / (0.5 + 0.5 * r) + 0.5 * 0.5 * r / (0.5 + 0.5 * r) = r / (1 + r)
        expected_nonequal_pair_rate = REPETITION_PENALIZE_COEFFICIENT / (1 + REPETITION_PENALIZE_COEFFICIENT)
        expected_nonequal_pair_rate_from, expected_nonequal_pair_rate_to = \
            binom.interval(1 - _CONFIDENCE_LEVEL, _SAMPLES_NUM, expected_nonequal_pair_rate)
        self.assertLessEqual(actual_num_nonequal_pairs, expected_nonequal_pair_rate_to)
        self.assertGreaterEqual(actual_num_nonequal_pairs, expected_nonequal_pair_rate_from)

    def test_nonpenalizable_tokens(self):
        probs = [0.5, 0.5]

        actual_num_nonequal_pairs = 0
        samples_generated = 0
        while samples_generated < _SAMPLES_NUM:
            token_sampler = TokenSampler(
                batch_size=1,
                banned_tokens_ids=[],
                non_penalizable_tokens_ids=[0],
                repetition_penalization_coefficient=REPETITION_PENALIZE_COEFFICIENT)
            first_token = token_sampler.sample(probs, sample_idx=0)
            if first_token == 0:
                samples_generated += 1
                second_token = token_sampler.sample(probs, sample_idx=0)
                actual_num_nonequal_pairs += (first_token != second_token)

        # When we don't penalize for token#0, P(first != second | first=0) = P(second=1 | first=0) = 0.5
        expected_nonequal_pair_rate = 0.5
        expected_nonequal_pair_rate_from, expected_nonequal_pair_rate_to = binom.interval(
            1 - _CONFIDENCE_LEVEL, _SAMPLES_NUM, expected_nonequal_pair_rate)
        self.assertLessEqual(actual_num_nonequal_pairs, expected_nonequal_pair_rate_to)
        self.assertGreaterEqual(actual_num_nonequal_pairs, expected_nonequal_pair_rate_from)

    def test_nonpenalizable_tokens_2(self):
        probs = [0.5, 0.5]

        actual_num_nonequal_pairs = 0
        samples_generated = 0
        while samples_generated < _SAMPLES_NUM:
            token_sampler = TokenSampler(
                batch_size=1,
                banned_tokens_ids=[],
                non_penalizable_tokens_ids=[1],
                repetition_penalization_coefficient=REPETITION_PENALIZE_COEFFICIENT)
            first_token = token_sampler.sample(probs, sample_idx=0)
            if first_token == 0:
                samples_generated += 1
                second_token = token_sampler.sample(probs, sample_idx=0)
                actual_num_nonequal_pairs += (first_token != second_token)

        # When we penalize for token#0, P(first != second | first=0) = P(second=1 | first=0) = 0.5 * r / (0.5 + 0.5 * r) = r / (1 + r)
        expected_nonequal_pair_rate = REPETITION_PENALIZE_COEFFICIENT / (1 + REPETITION_PENALIZE_COEFFICIENT)
        expected_nonequal_pair_rate_from, expected_nonequal_pair_rate_to = binom.interval(
            1 - _CONFIDENCE_LEVEL, _SAMPLES_NUM, expected_nonequal_pair_rate)
        self.assertLessEqual(actual_num_nonequal_pairs, expected_nonequal_pair_rate_to)
        self.assertGreaterEqual(actual_num_nonequal_pairs, expected_nonequal_pair_rate_from)


if __name__ == '__main__':
    unittest.main()
