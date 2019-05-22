from abc import ABCMeta, abstractmethod
from itertools import zip_longest

import numpy as np

from cakechat.dialog_model.inference.service_tokens import ServiceTokensIDs
from cakechat.dialog_model.inference.utils import get_sequence_score_by_thought_vector, get_sequence_score, \
    get_thought_vectors
from cakechat.dialog_model.model_utils import reverse_nn_input
from cakechat.utils.data_types import Dataset
from cakechat.utils.logger import get_logger
from cakechat.utils.profile import timer

_logger = get_logger(__name__)


class AbstractCandidatesReranker(object, metaclass=ABCMeta):
    @abstractmethod
    def rerank_candidates(self, contexts, all_candidates, condition_ids):
        pass


class DummyReranker(AbstractCandidatesReranker):
    def rerank_candidates(self, contexts, all_candidates, condition_ids):
        return all_candidates


class MMIReranker(AbstractCandidatesReranker):
    """
    Ranks candidates based on the MMI-score:
     score = (1 - \lambda) log p(y|x) + \lambda log p(x|y) - \beta R_y,
    where
     - x is dialogue context;
     - y is a candidate response;
     - R_y is the number of repeated tokens used in a candidate response
     - \lambda, \beta - hyperparameters. \beta = log(REPETITION_PENALIZE_COEFFICIENT)

    Score formula is based on (9) https://arxiv.org/pdf/1510.03055v3.pdf
    """

    def __init__(self, nn_model, reverse_model, mmi_reverse_model_score_weight, repetition_penalization_coefficient):
        self._nn_model = nn_model
        if mmi_reverse_model_score_weight != 0.0 and reverse_model is None:
            raise ValueError('Reverse model has to be supplied to MMI-reranker. '
                             'If you don\'t have one, set mmi_reverse_model_score_weight to 0.')
        self._reverse_model_score_weight = mmi_reverse_model_score_weight
        self._reverse_model = reverse_model
        self._service_tokens_ids = ServiceTokensIDs(nn_model.token_to_index)
        self._log_repetition_penalization_coefficient = np.log(repetition_penalization_coefficient)

    def _compute_likelihood_of_output_given_input(self, thought_vector, candidates, condition_id):
        # Repeat to get same thought vector for each candidate
        thoughts_batch = np.repeat(thought_vector, candidates.shape[0], axis=0)
        return get_sequence_score_by_thought_vector(self._nn_model, thoughts_batch, candidates, condition_id)

    def _compute_likelihood_of_input_given_output(self, context, candidates, condition_id):
        # Repeat to get same context for each candidate
        repeated_context = np.repeat(context, candidates.shape[0], axis=0)
        reversed_dataset = reverse_nn_input(
            Dataset(x=repeated_context, y=candidates, condition_ids=None), self._service_tokens_ids)
        return get_sequence_score(self._reverse_model, reversed_dataset.x, reversed_dataset.y, condition_id)

    def _compute_num_repetitions(self, candidates):
        skip_tokens_ids = \
            self._service_tokens_ids.special_tokens_ids + self._service_tokens_ids.non_penalizable_tokens_ids
        result = []
        for candidate in candidates:
            penalizable_tokens = candidate[~np.in1d(candidate, skip_tokens_ids)]  # All tokens not in skip_tokens_ids
            num_repetitions = penalizable_tokens.size - np.unique(penalizable_tokens).size
            result.append(num_repetitions)
        return np.array(result)

    def _compute_candidates_scores(self, context, candidates, condition_id):
        context = context[np.newaxis, :]  # from (seq_len,) to (1 x seq_len)
        thought_vector = get_thought_vectors(self._nn_model, context)

        candidates_num_repetitions = self._compute_num_repetitions(candidates)

        if self._reverse_model_score_weight == 0.0:
            candidates_scores = self._compute_likelihood_of_output_given_input(thought_vector, candidates, condition_id)
        elif self._reverse_model_score_weight == 1.0:  # Don't compute the likelihood in this case for performance
            candidates_scores = self._compute_likelihood_of_input_given_output(context, candidates, condition_id)
        else:
            candidates_likelihood = self._compute_likelihood_of_output_given_input(thought_vector, candidates,
                                                                                   condition_id)
            candidates_reverse_likelihood = self._compute_likelihood_of_input_given_output(
                context, candidates, condition_id)
            candidates_scores = (1 - self._reverse_model_score_weight) * candidates_likelihood + \
                                self._reverse_model_score_weight * candidates_reverse_likelihood

        candidates_scores -= self._log_repetition_penalization_coefficient * candidates_num_repetitions
        return candidates_scores

    @timer
    def rerank_candidates(self, contexts, all_candidates, condition_ids):
        condition_ids = [] if condition_ids is None else condition_ids  # For izip_lingest
        candidates_scores = [
            self._compute_candidates_scores(context, candidates, condition_id)
            for context, candidates, condition_id in zip_longest(contexts, all_candidates, condition_ids)
        ]
        scores_order = [np.argsort(-np.array(scores)) for scores in candidates_scores]
        batch_size = len(contexts)
        # reranked_candidates[i][j] = j-th best response for i-th question
        reranked_candidates = [
            [all_candidates[i][j] for j in scores_order[i]] for i in range(batch_size)  # yapf: disable
        ]
        return reranked_candidates
