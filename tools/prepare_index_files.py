import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import codecs
from collections import Counter
import json

from cakechat.utils.files_utils import is_non_empty_file, ensure_dir
from cakechat.utils.text_processing import get_tokens_sequence, get_processed_corpus_path, get_index_to_token_path, \
    get_index_to_condition_path, load_processed_dialogs_from_json, FileTextLinesIterator, SPECIAL_TOKENS
from cakechat.config import BASE_CORPUS_NAME, TRAIN_CORPUS_NAME, DEFAULT_CONDITION

MAX_TOKENS_NUM = 50000
MAX_CONDITIONS_NUM = 5
TEXT_FIELD_NAME = 'text'
CONDITION_FIELD_NAME = 'condition'


def build_index_mappings(corpus_path, max_tokens_num=MAX_TOKENS_NUM, max_conditions_num=MAX_CONDITIONS_NUM):
    if not is_non_empty_file(corpus_path):
        raise ValueError('Test corpus file doesn\'t exist: {}'.format(corpus_path))

    dialogs = load_processed_dialogs_from_json(
        FileTextLinesIterator(corpus_path), text_field_name=TEXT_FIELD_NAME, condition_field_name=CONDITION_FIELD_NAME)

    tokens_counter = Counter()
    conditions_counter = Counter()

    for dialog in dialogs:
        for utterance in dialog:
            # Tokenize dialog utterance text and update tokens count
            tokens = get_tokens_sequence(utterance[TEXT_FIELD_NAME])
            tokens_counter += Counter(tokens)
            # Update conditions count
            conditions_counter[utterance[CONDITION_FIELD_NAME]] += 1

    # Build the tokens list
    vocab = list(SPECIAL_TOKENS) + \
            [token for token, _ in tokens_counter.most_common(max_tokens_num - len(SPECIAL_TOKENS))]

    # Build the conditions list
    conditions = [condition for condition, _ in conditions_counter.most_common(max_conditions_num)]

    # Validate the condition list
    if DEFAULT_CONDITION not in conditions:
        raise Exception('No default condition "%s" found in the dataset condition list.' % DEFAULT_CONDITION)

    # Return index_to_token and index_to_condition mappings
    return dict(enumerate(vocab)), dict(enumerate(conditions))


def dump_index_to_item(index_to_item, path):
    ensure_dir(os.path.dirname(path))
    with codecs.open(path, 'w', 'utf-8') as fh:
        json.dump(index_to_item, fh, ensure_ascii=False)


if __name__ == '__main__':
    processed_train_corpus_path = get_processed_corpus_path(TRAIN_CORPUS_NAME)
    index_to_token_path = get_index_to_token_path(BASE_CORPUS_NAME)
    index_to_condition_path = get_index_to_condition_path(BASE_CORPUS_NAME)

    index_to_token, index_to_condition = build_index_mappings(processed_train_corpus_path)
    dump_index_to_item(index_to_token, index_to_token_path)
    dump_index_to_item(index_to_condition, index_to_condition_path)
