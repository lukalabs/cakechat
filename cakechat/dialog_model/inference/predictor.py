import numpy as np
from six.moves import xrange


class Predictor(object):
    def __init__(self, nn_model, candidates_generator, reranker):
        self._nn_model = nn_model
        self._generator = candidates_generator
        self._reranker = reranker

    @staticmethod
    def _select_best_candidates(reranked_candidates, candidates_num):
        """
        We need this complicated implementation to handle different number of generated candidates for each sample.
        If for some context we generated less then candidates_num candidates, we fill this responses with pads.
        """
        batch_size = len(reranked_candidates)
        # reranked_candidates is list of lists (we need too keep it this way because we can have different number
        # of candidates for each context), so we can't just write rerankied_candidates.shape[2]
        output_seq_len = reranked_candidates[0][0].size
        result = np.zeros((batch_size, candidates_num, output_seq_len))
        # Loop here instead of slices because number of candidates for each context can vary here
        for i in xrange(batch_size):
            for j, candidate in enumerate(reranked_candidates[i]):
                if j >= candidates_num:
                    break
                result[i][j] = reranked_candidates[i][j]
        return result

    def predict_responses(self, context_token_ids, output_seq_len, condition_ids=None, candidates_num=1):
        all_candidates = self._generator.generate_candidates(context_token_ids, condition_ids, output_seq_len)
        reranked_candidates = self._reranker.rerank_candidates(context_token_ids, all_candidates, condition_ids)
        return self._select_best_candidates(reranked_candidates, candidates_num)
