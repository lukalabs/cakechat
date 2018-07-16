import numpy as np
from six.moves import xrange, zip_longest
import theano

from cakechat.dialog_model.inference.candidates.abstract_generator import AbstractCandidatesGenerator
from cakechat.dialog_model.inference.service_tokens import ServiceTokensIDs
from cakechat.dialog_model.inference.utils import get_next_token_log_prob_one_step, get_thought_vectors
from cakechat.utils.profile import timer


class BeamsearchCandidatesGenerator(AbstractCandidatesGenerator):
    def __init__(self, nn_model, beam_size, repetition_penalization_coefficient):
        """
        :param nn_model: NN model to use for predicting
        :param beam_size: Size of beam
        """
        self._nn_model = nn_model
        self._beam_size = min(beam_size, self._nn_model.vocab_size)
        self._num_finished_candidates_to_keep = self._beam_size ** 2
        self._log_repetition_penalization_coefficient = np.log(repetition_penalization_coefficient)
        self._service_tokens_ids = ServiceTokensIDs(nn_model.token_to_index)

    @staticmethod
    def _get_k_max_elements_indices_and_scores(vec, k, mask=None):
        if mask is None:
            # We use argpartition here instead of argsort to achieve linear-time performance.
            max_elements_indices = np.argpartition(-vec, k - 1)[:k]
        else:
            masked_vec = vec.copy()  # To avoid side-effects
            masked_vec[~mask] = -np.inf
            max_elements_indices = np.argpartition(-masked_vec, k - 1)[:k]
        return max_elements_indices, vec[max_elements_indices]

    def _init_hidden_states_and_candidates(self, output_seq_len):
        # This array will contain beam_size candidates, each of which output_seq_len long.
        # dtype=np.int32, because this array stores ids of tokens which are integers.
        self._cur_candidates = np.full(
            (self._beam_size, output_seq_len), self._service_tokens_ids.pad_token_id, dtype=np.int32)
        # First, fill in first token of each candidate
        self._cur_candidates[:, 0] = self._service_tokens_ids.start_token_id
        # and prepare an array for score of each candidate
        self._cur_candidates_scores = np.zeros(self._beam_size)

        # Finished candidates are going to be concatenated here. Each candidate will be of shape=(1, output_seq_len),
        # But now we have 0 candidates: that's why we need to initialize this array with shape=(0, output_seq_len)
        self._finished_candidates = np.zeros((0, output_seq_len), dtype=np.int32)
        # Same story here: 0 candidates so long => shape=0.
        self._finished_candidates_scores = np.zeros(0, dtype=np.float32)

        # For each candidate in the beam, for each layer of the decoder we need hidden_states_dim numbers to store
        # this array
        self._hidden_states_batch = np.zeros(
            (self._beam_size, self._nn_model.decoder_depth, self._nn_model.hidden_layer_dim),
            dtype=theano.config.floatX)  # By default, numpy has dtype=np.float64, but this array is passed
        # right into theano functions, so we need to have explicit type declaring here.

    def _compute_thought_vectors(self, context_token_ids):
        # thought_vector is (1 x thought_vector_dim);
        # context_token_ids is (input_seq_len), but we need to make it 1 x input_seq_len, because theano functions
        # require input_seq_len dimension to be the second one.
        thought_vector = get_thought_vectors(self._nn_model, context_token_ids[np.newaxis, :])
        # All theano functions process each sequence independently: every input sequence is matched to the corresponding
        # output sequence. So if we want to have probability of all outputs given the save inputs, we need to repeat
        # the input <num_outputs> times. <num_outputs> = beam_size here.
        self._thought_batch = np.repeat(thought_vector, self._beam_size, axis=0)

    def _update_next_candidates_and_hidden_states(self, token_idx, best_non_finished_candidates_indices,
                                                  expanded_beam_tokens):
        """
        Updates current state of candidates prediction process and fills in current candidates.

        :param token_idx: position of current token
        :param expanded_beam_tokens: np.array with shape (beam_size * beam_size,)
            Tokens candidates for the next step.
        :param best_non_finished_candidates_indices: np.array with shape (beam_size,)
            Contains indexes of best K candidates in current K^2 sized expanded beam to use in the next beam.
        """
        # Separate arrays for the updated hidden states
        next_hidden_states_batch = np.zeros_like(self._hidden_states_batch)
        # and the candidates
        next_step_candidates = np.full_like(self._cur_candidates, self._service_tokens_ids.pad_token_id, dtype=np.int32)

        for i, candidate_idx in enumerate(best_non_finished_candidates_indices):
            # expanded_beam_tokens contains the last token for each of the beam_size^2 candidates in the expanded beam.
            # We need to get which original candidate this token in the expanded beam corresponds to.
            # (to fill in all the previous tokens from self._cur_candidates)
            # Because all the candidates in the expanded beam were filled sequentially, we just use this formula:
            original_candidate_idx = candidate_idx // self._beam_size

            # Construct the candidates for the next step using self._cur_candidates and the last token:

            # next_tokens is the last token for each new candidate here.
            next_token = expanded_beam_tokens[candidate_idx]

            # First, fill in all the preceding tokens for current candidate
            next_step_candidates[i, :token_idx] = self._cur_candidates[original_candidate_idx, :token_idx]
            # And put the last token of the current candidate on its position
            next_step_candidates[i, token_idx] = next_token
            # We also have to update the hidden states for the next step: we need to know which hidden states
            # to use to continue decoding each candidate and we get them from the corresponding positions
            next_hidden_states_batch[i] = self._hidden_states_batch[original_candidate_idx]

        self._hidden_states_batch = next_hidden_states_batch
        self._cur_candidates = next_step_candidates

    def _update_finished_candidates(self, token_idx, best_finished_candidates_indices, expanded_beam_scores,
                                    expanded_beam_tokens, output_seq_len):
        n_finished_candidates = len(best_finished_candidates_indices)
        if not n_finished_candidates:
            return

        # These are only finished candidates on the current step. We will further append this array to
        # self._finished_candidates
        # dtype=np.int32, because this array stores ids of tokens which are integers.
        cur_finished_candidates = \
            np.full((n_finished_candidates, output_seq_len), self._service_tokens_ids.pad_token_id, dtype=np.int32)

        cur_finished_candidates_scores = np.full(n_finished_candidates, 0, dtype=np.float32)
        for i, candidate_idx in enumerate(best_finished_candidates_indices):
            # expanded_beam_tokens contains the last token for each of the beam_size^2 candidates in the expanded beam
            # to get all the other tokens we need to get which original candidate this token in the expanded beam
            # corresponds to. Because all the candidates in the expanded beam were filled sequentially, we can just
            # use this formula:
            original_candidate_idx = candidate_idx // self._beam_size

            # Construct the candidates for the next step using self._cur_candidates and the last token:

            # next_tokens is the last token for each new candidate here.
            next_token = expanded_beam_tokens[candidate_idx]
            # Also we do the same thing for scores
            candidate_score = expanded_beam_scores[candidate_idx]

            # First, fill in all the preceding tokens for current candidate
            cur_finished_candidates[i, :token_idx] = self._cur_candidates[original_candidate_idx, :token_idx]
            # And put the last token of the current candidate on its position
            cur_finished_candidates[i, token_idx] = next_token
            cur_finished_candidates_scores[i] = candidate_score

        # Use concatenate to add sequences of the same length to the list of the sequences.
        # Use axis=0 here, because 0-th dimension corresponds to the index of the candidate in the list
        # And 1-st dimension enumerates the tokens within each sequence.
        self._finished_candidates = np.concatenate((self._finished_candidates, cur_finished_candidates), axis=0)
        self._finished_candidates_scores = np.concatenate(
            (self._finished_candidates_scores, cur_finished_candidates_scores), axis=0)

    def _penalize_by_repetition(self, next_token_log_probs_batch, used_tokens_ids_batch):
        for i, used_tokens_ids in enumerate(used_tokens_ids_batch):
            tokens_ids_to_penalize = [
                x for x in used_tokens_ids if x not in self._service_tokens_ids.non_penalizable_tokens_ids
            ]
            next_token_log_probs_batch[i, tokens_ids_to_penalize] -= self._log_repetition_penalization_coefficient
        return next_token_log_probs_batch

    def _compute_next_token_score_batch(self, token_idx, condition_id):
        # Get prediction of the model - p(T|S)
        current_token_id_for_each_candidate = self._cur_candidates[:, token_idx - 1]
        self._hidden_states_batch, next_token_score_batch = \
            get_next_token_log_prob_one_step(self._nn_model, self._thought_batch, self._hidden_states_batch,
                                             current_token_id_for_each_candidate, condition_id)
        # Candidates[:, :token_idx], as usual, means "All tokens preceding token_idx for each candidate"
        # We use all preceding tokens to compute counts and penalize current distribution using these counts.
        next_token_score_batch = self._penalize_by_repetition(next_token_score_batch,
                                                              self._cur_candidates[:, :token_idx])

        # next_token_score_batch here is (num_candidates x vocab_size). For each candidate we find all the
        # banned tokens and kill their scores to -inf.
        next_token_score_batch[:, self._service_tokens_ids.banned_tokens_ids] = -np.inf
        return next_token_score_batch

    def _get_aggregated_scores_and_tokens_for_expanded_beam(self, next_token_score_batch):
        # Expanded beam is beam_size x beam_size candidates that we consider on the next step.
        # But we don't want to keep all the candidates themselves for better performance. So we just keep
        # the last token and the total score.
        expanded_beam_scores = np.zeros((self._beam_size * self._beam_size), dtype=np.float32)
        expanded_beam_tokens = np.zeros((self._beam_size * self._beam_size), dtype=np.int32)

        for candidate_idx in xrange(self._beam_size):
            # Get beam_size candidates on each step
            next_token_candidates, next_token_scores = \
                self._get_k_max_elements_indices_and_scores(next_token_score_batch[candidate_idx], self._beam_size)
            # sequentially fill in the additive scores
            expanded_beam_scores[candidate_idx * self._beam_size:(candidate_idx + 1) * self._beam_size] = \
                self._cur_candidates_scores[candidate_idx] + next_token_scores
            # and the corresponding last tokens in the array
            expanded_beam_tokens[candidate_idx * self._beam_size:(candidate_idx + 1) * self._beam_size] = \
                next_token_candidates
        return expanded_beam_scores, expanded_beam_tokens

    def _get_best_finished_and_nonfinished_candidates(self, expanded_beam_scores, expanded_beam_tokens):
        """
        Get top-k next tokens for each candidate.
        Also updates aggregated scores according to the scores for chosen tokens.

        :param expanded_beam_scores: Aggregated scores for each response in the extended beam
        :param expanded_beam_tokens: Last tokens in each response in the extended beam
        :return: best_non_finished_candidates, best_finished_candidates
            best_non_finished_candidates are used for updating beam for the next step
            best_finished_candidates are returned to dump on each step and rerank afterwards.
        """
        # This mask contains true if the corresponding candidate is finished with <EOS> and false otherwise
        finished_candidates_mask = (expanded_beam_tokens == self._service_tokens_ids.eos_token_id)
        # We select the best candidates among those who are not finished
        best_non_finished_candidates, self._cur_candidates_scores = \
            self._get_k_max_elements_indices_and_scores(expanded_beam_scores, self._beam_size,
                                                        ~finished_candidates_mask)
        # And also return finished candidates that are good enough to fit in the new beam, but do not go there
        high_quality_candidates_mask = (expanded_beam_scores > self._cur_candidates_scores[self._beam_size - 1] - 1e-6)
        # We need [0] in the end of this line because np.nonzero returns tuple, but we only need indices
        best_finished_candidates = np.nonzero(finished_candidates_mask & high_quality_candidates_mask)[0]
        return best_non_finished_candidates, best_finished_candidates

    def _generate_candidates_for_one_context(self, condition_id, output_seq_len):
        # Fill the first beam_size candidates.
        # We need to do it separately here, because the logic is a little bit different from what is going on for all
        # the other steps. There we generate beam_size next tokens for each each current candidate in the beam and then
        # select the beam_size best ones. But here we just compute the initial beam_size candidates according to
        # the score of the 1-st token.
        next_token_score_batch = self._compute_next_token_score_batch(1, condition_id)
        self._cur_candidates[:, 1], self._cur_candidates_scores = self._get_k_max_elements_indices_and_scores(
            next_token_score_batch[0], self._beam_size)

        for token_idx in xrange(2, output_seq_len):  # Start from 2 because first token candidates are already filled.
            # This array has shape beam_size x vocab_size. We use this scores to select best tokens for the beam
            # on the next step.
            next_token_score_batch = self._compute_next_token_score_batch(token_idx, condition_id)

            # Select beam_size best tokens for each candidate in the beam.
            # Also compute the score of the corresponding candidate.
            expanded_beam_scores, expanded_beam_tokens = \
                self._get_aggregated_scores_and_tokens_for_expanded_beam(next_token_score_batch)

            # Select the best candidates according to the scores computed prevuoisly
            best_non_finished_candidates_indices, best_finished_candidates_indices = \
                self._get_best_finished_and_nonfinished_candidates(expanded_beam_scores, expanded_beam_tokens)

            self._update_finished_candidates(token_idx, best_finished_candidates_indices, expanded_beam_scores,
                                             expanded_beam_tokens, output_seq_len)
            self._update_next_candidates_and_hidden_states(token_idx, best_non_finished_candidates_indices,
                                                           expanded_beam_tokens)

        # Pre-filter candidates based on intermidiate scores for better performance
        output_candidates_num = min(self._beam_size, self._finished_candidates.shape[0])
        idxs, _ = self._get_k_max_elements_indices_and_scores(self._finished_candidates_scores, output_candidates_num)
        self._finished_candidates = self._finished_candidates[idxs]

        return self._finished_candidates if self._finished_candidates.shape[0] > 0 else self._cur_candidates

    @timer
    def generate_candidates(self, context_token_ids, condition_ids, output_seq_len):
        x_with_conditions_batch = zip_longest(context_token_ids, condition_ids if condition_ids is not None else [])
        result = []
        for x, condition_id in x_with_conditions_batch:
            self._compute_thought_vectors(x)
            self._init_hidden_states_and_candidates(output_seq_len)
            result.append(self._generate_candidates_for_one_context(condition_id, output_seq_len))
        return result
