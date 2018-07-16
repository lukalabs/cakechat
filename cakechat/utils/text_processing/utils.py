import os
import json
import codecs

from six import iteritems

from cakechat.config import PROCESSED_CORPUS_DIR, TOKEN_INDEX_DIR, CONDITION_IDS_INDEX_DIR


def get_processed_corpus_path(corpus_name):
    return os.path.join(PROCESSED_CORPUS_DIR, corpus_name + '.txt')


def get_index_to_token_path(processed_corpus_name):
    return os.path.join(TOKEN_INDEX_DIR, 't_idx_{}.json'.format(processed_corpus_name))


def get_index_to_condition_path(processed_corpus_name):
    return os.path.join(CONDITION_IDS_INDEX_DIR, 'c_idx_{}.json'.format(processed_corpus_name))


def load_index_to_item(items_index_path):
    with codecs.open(items_index_path, 'r', 'utf-8') as item_index_fh:
        index_to_item = json.load(item_index_fh)
        index_to_item = {int(k): v for k, v in iteritems(index_to_item)}

    return index_to_item
