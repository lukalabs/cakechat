import nltk

from cakechat.utils.data_structures import flatten
from cakechat.utils.files_utils import load_file
from cakechat.utils.text_processing import get_tokens_sequence


class OffenseDetector(object):
    def __init__(self, offensive_phrases_path):
        self._offensive_ngrams = self._build_offensive_ngrams(offensive_phrases_path)
        self._max_ngram_len = max(map(len, self._offensive_ngrams))

    @property
    def offensive_ngrams(self):
        return self._offensive_ngrams

    @staticmethod
    def _build_offensive_ngrams(offensive_phrases_path):
        offensive_phrases = load_file(offensive_phrases_path)
        offensive_ngrams = [tuple(get_tokens_sequence(offensive_phrase)) for offensive_phrase in offensive_phrases]
        return set(offensive_ngrams)

    def _get_ngrams(self, tokenized_line):
        ngrams = [nltk.ngrams(tokenized_line, i) for i in range(1, self._max_ngram_len + 1)]
        return flatten(ngrams, constructor=set)

    def has_offensive_ngrams(self, text):
        if not isinstance(text, str):
            raise TypeError('"text" variable must be a string')
        tokenized_text = get_tokens_sequence(text)
        text_ngrams = self._get_ngrams(tokenized_text)

        return bool(text_ngrams & self._offensive_ngrams)
