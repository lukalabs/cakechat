from abc import ABCMeta, abstractmethod


class AbstractCandidatesGenerator(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def generate_candidates(self, context_token_ids, condition_ids, output_seq_len):
        pass
