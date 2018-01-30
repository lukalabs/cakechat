from cakechat.config import NON_PENALIZABLE_TOKENS
from cakechat.utils.offense_detector.config import OFFENSIVE_PHRASES_PATH
from cakechat.utils.offense_detector import OffenseDetector
from cakechat.utils.text_processing import SPECIAL_TOKENS

_offense_detector = OffenseDetector(OFFENSIVE_PHRASES_PATH)


class ServiceTokensIDs(object):
    """
    Handles computation of indices of all special tokens needed for predicting and reranking of responses
    """

    def __init__(self, token_to_index):
        self.start_token_id = token_to_index[SPECIAL_TOKENS.START_TOKEN]
        self.eos_token_id = token_to_index[SPECIAL_TOKENS.EOS_TOKEN]
        self.pad_token_id = token_to_index[SPECIAL_TOKENS.PAD_TOKEN]
        self.unk_token_id = token_to_index[SPECIAL_TOKENS.UNKNOWN_TOKEN]
        self.special_tokens_ids = [self.start_token_id, self.eos_token_id, self.pad_token_id, self.unk_token_id]

        # Get first token for each offensive ngram
        offensive_tokens = [ngram[0] for ngram in _offense_detector.offensive_ngrams if len(ngram) == 1]
        # We don't penalize for repeating these tokens:
        self.non_penalizable_tokens_ids = [token_to_index[w] for w in NON_PENALIZABLE_TOKENS if w in token_to_index]
        # These tokens are banned during the prediction:
        offensive_tokens_ids = [token_to_index[w] for w in offensive_tokens if w in token_to_index]
        self.banned_tokens_ids = offensive_tokens_ids + [self.unk_token_id]
