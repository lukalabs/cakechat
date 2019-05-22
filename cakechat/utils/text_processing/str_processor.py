import re

import nltk.tokenize

from cakechat.utils.text_processing.config import SPECIAL_TOKENS

_END_CHARS = '.?!'

_tokenizer = nltk.tokenize.RegexpTokenizer(pattern='\w+|[^\w\s]')


def get_tokens_sequence(text, lower=True, check_unicode=True):
    if check_unicode and not isinstance(text, str):
        raise TypeError('Text object should be unicode type. Got instead "{}" of type {}'.format(text, type(text)))

    if not text.strip():
        return []

    if lower:
        text = text.lower()

    tokens = _tokenizer.tokenize(text)

    return tokens


def replace_out_of_voc_tokens(tokens, tokens_voc):
    return [t if t in tokens_voc else SPECIAL_TOKENS.UNKNOWN_TOKEN for t in tokens]


def _capitalize_first_chars(text):
    if not text:
        return text

    chars_pos_to_capitalize = [0] + [m.end() - 1 for m in re.finditer('[{}] \w'.format(_END_CHARS), text)]

    for char_pos in chars_pos_to_capitalize:
        text = text[:char_pos] + text[char_pos].upper() + text[char_pos + 1:]

    return text


def prettify_response(response):
    """
    Prettify chatbot's answer removing excessive characters and capitalizing first words of sentences.
    Before: "hello world ! nice to meet you , buddy . do you like me ? I ' ve been missing you for a while . . ."
    After: "Hello world! Nice to meet you, buddy. Do you like me? I've been missing you for a while..."
    """
    response = response.replace(' \' ', '\'')

    for ch in set(_END_CHARS) | {','}:
        response = response.replace(' ' + ch, ch)

    response = _capitalize_first_chars(response)
    response = response.strip()

    return response
