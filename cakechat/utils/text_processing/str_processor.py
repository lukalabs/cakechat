import re

import nltk.tokenize

from cakechat.utils.text_processing.config import SPECIAL_TOKENS

_END_CHARS = '.?!'

_tokenizer = nltk.tokenize.RegexpTokenizer(pattern=ur'\w+|[^\w\s]')


def get_tokens_sequence(text, lower=True, check_unicode=True):
    if check_unicode and not isinstance(text, unicode):
        raise TypeError('text object should be unicode type')

    if not text.strip():
        return []

    if lower:
        text = text.lower()

    tokens = _tokenizer.tokenize(text)

    return tokens


def replace_out_of_voc_tokens(tokens, tokens_voc):
    return map(lambda t: t if t in tokens_voc else SPECIAL_TOKENS.UNKNOWN_TOKEN, tokens)


def _capitalize_first_chars(text):
    if not text:
        return text

    chars_pos_to_capitalize = [0] + [m.end() - 1 for m in re.finditer('[%s] \w' % _END_CHARS, text)]

    for char_pos in chars_pos_to_capitalize:
        text = text[:char_pos] + text[char_pos].upper() + text[char_pos + 1:]

    return text


def get_pretty_str_from_tokens_sequence(tokens_sequence):
    """
    Prettify chatbot's answer removing excessive characters and capitalizing first words of sentences.
    Before: "hello world ! nice to meet you , buddy . do you like me ? I ' ve been missing you for a while . . . $$$"
    After: "Hello world! Nice to meet you, buddy. Do you like me? I've been missing you for a while..."
    """
    phrase = ' '.join(tokens_sequence)

    phrase = phrase.replace(SPECIAL_TOKENS.EOS_TOKEN, '')
    phrase = phrase.replace(SPECIAL_TOKENS.START_TOKEN, '')
    phrase = phrase.replace(' \' ', '\'')

    for ch in set(_END_CHARS) | {','}:
        phrase = phrase.replace(' ' + ch, ch)

    phrase = _capitalize_first_chars(phrase)
    phrase = phrase.strip()

    return phrase
