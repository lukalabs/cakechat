import numpy as np
from six.moves import xrange
import theano

from cakechat.dialog_model.inference.candidates.abstract_generator import AbstractCandidatesGenerator
from cakechat.dialog_model.inference.service_tokens import ServiceTokensIDs
from cakechat.dialog_model.inference.utils import get_next_token_prob_one_step, get_thought_vectors


class TokenSampler(object):
    """
    Class for sampling responses without banned tokens and repeating.

    There has to be individual instance of TokenSampler for each run of sampling procedure
    because it contains counters of tokens that have to be reset before sampling new responses.
    """

    def __init__(self, batch_size, banned_tokens_ids, non_penalizable_tokens_ids, repetition_penalization_coefficient):
        self._batch_size = batch_size
        self._banned_tokens_ids = banned_tokens_ids
        self._non_penalizable_tokens_ids = non_penalizable_tokens_ids
        self._used_tokens_ids = [[] for _ in xrange(batch_size)]
        self._repetition_penalization_coefficient = repetition_penalization_coefficient

    def sample(self, probabilities, sample_idx, temperature=1.0):
        """
        Sample using individual priors for each sample_idx.
        Also updates individual priors for each sample in batch.
        We need individual priors to prevent the model from repeating the same tokens over and over again in one
        response.

        probabilities: Probabilities of each token. The distribution given by these probabilities
            is used by this function to sample the token.
        sample_idx : Integer between 0 and batch_size-1. We need it to figure out which token_log_prior line to use.
        temperature: Temperature for sampling. Temperature has to be a positive number.
        :return: Index of sampled token
        """
        # To make the repetition penalization invariant to the original temperature we have to adjust the coefficient:
        repetition_penalize_coefficient = np.exp(np.log(self._repetition_penalization_coefficient) / temperature)
        # Back-up the array to avoid side-effects (otherwise the function will change the probabilities passed as an
        # argument)
        probabilities = np.copy(probabilities)

        probabilities[self._banned_tokens_ids] = 0
        probabilities[self._used_tokens_ids[sample_idx]] /= repetition_penalize_coefficient

        probabilities /= np.sum(probabilities)
        token_id = np.random.choice(probabilities.shape[0], replace=False, p=probabilities)

        # Update used tokens list
        if token_id not in self._non_penalizable_tokens_ids:
            self._used_tokens_ids[sample_idx].append(token_id)

        return token_id


class SamplingCandidatesGenerator(AbstractCandidatesGenerator):
    def __init__(self, nn_model, temperature, samples_num, repetition_penalization_coefficient):
        self._nn_model = nn_model
        self._temperature = temperature
        self._samples_num = samples_num
        self._service_tokens_ids = ServiceTokensIDs(nn_model.token_to_index)
        self._repetition_penalization_coefficient = repetition_penalization_coefficient

    def _sample_response(self, thought_vectors, condition_ids, output_seq_len):
        batch_size = thought_vectors.shape[0]
        sampler = TokenSampler(batch_size, self._service_tokens_ids.banned_tokens_ids,
                               self._service_tokens_ids.non_penalizable_tokens_ids,
                               self._repetition_penalization_coefficient)
        # For each candidate in the batch, for each layer of the decoder we need hidden_states_dim numbers to store
        # this array
        hidden_states_batch = np.zeros(
            (batch_size, self._nn_model.decoder_depth, self._nn_model.hidden_layer_dim),
            dtype=theano.config.floatX)  # By default, numpy has dtype=np.float64, but this array is passed
        # right into theano functions, so we need to have explicit type declaring here.

        response_tokens_ids = np.full(
            (batch_size, output_seq_len), self._service_tokens_ids.pad_token_id, dtype=np.int32)

        # Track finished responses to skip prediction step for them
        is_response_finished = np.zeros(batch_size, dtype=np.bool)

        # Fill in first tokens of each response in the batch:
        response_tokens_ids[:, 0] = self._service_tokens_ids.start_token_id
        for token_idx in xrange(1, output_seq_len):  # Starting with the second token
            hidden_states_batch, next_token_probs_batch = \
                get_next_token_prob_one_step(self._nn_model, thought_vectors, hidden_states_batch,
                                             response_tokens_ids[:, token_idx - 1],  # previous token for each response
                                             condition_ids,
                                             temperature=self._temperature)

            for response_idx, next_token_probs in enumerate(next_token_probs_batch):
                if is_response_finished[response_idx]:
                    continue

                next_token_id = sampler.sample(next_token_probs, response_idx, self._temperature)
                response_tokens_ids[response_idx, token_idx] = next_token_id

                if next_token_id in [self._service_tokens_ids.eos_token_id, self._service_tokens_ids.pad_token_id]:
                    is_response_finished[response_idx] = True

            # Stop if all responses are done
            if np.all(is_response_finished):
                break

        return response_tokens_ids

    def generate_candidates(self, context_tokens_ids, condition_ids, output_seq_len):
        """
        Predict answers for every sequence token by token until EOS_TOKEN occurred in the sequence
        using sampling with temperature.
        During the sampling procedure offensive and <unk> tokens are banned.
        Probabilities of tokens that have already been used in a response are penalized
        (divided by REPETITION_PENALIZE_COEFFICIENT).
        All the rest of the sequence is filled with PAD_TOKENs.
        """
        thought_vectors = get_thought_vectors(self._nn_model, context_tokens_ids)
        sampled_candidates = [
            self._sample_response(thought_vectors, condition_ids, output_seq_len) for _ in xrange(self._samples_num)
        ]

        # Transpose the result: candidate_id x batch_size x seq_len -> batch_size x candidate_id x seq_len
        return np.swapaxes(sampled_candidates, 0, 1)
