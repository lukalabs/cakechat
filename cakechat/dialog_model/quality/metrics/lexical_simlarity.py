import os

from sklearn.feature_extraction.text import TfidfVectorizer

from cakechat.config import TRAIN_CORPUS_NAME, BASE_CORPUS_NAME, DATA_DIR
from cakechat.utils.files_utils import get_persisted
from cakechat.utils.text_processing import load_index_to_item, get_tokens_sequence, get_processed_corpus_path, \
    load_processed_dialogs_from_json, FileTextLinesIterator, get_dialog_lines_and_conditions, \
    get_alternated_dialogs_lines, get_index_to_token_path

_TFIDF_VECTORIZER_FULL_PATH = os.path.join(DATA_DIR, 'tfidf_vectorizer.pickle')


def _load_train_lines(corpus_name=TRAIN_CORPUS_NAME):
    processed_corpus_path = get_processed_corpus_path(corpus_name)
    dialogs = load_processed_dialogs_from_json(
        FileTextLinesIterator(processed_corpus_path), text_field_name='text', condition_field_name='condition')
    train_lines, _ = get_dialog_lines_and_conditions(
        get_alternated_dialogs_lines(dialogs), text_field_name='text', condition_field_name='condition')
    return train_lines


def _calculate_tfidf_vector(tfidf_vectorizer, y_sequence):
    # tfidf-transformer works with list of items,
    # therefore we should supply list of sequences and then slice the first element.
    return tfidf_vectorizer.transform([y_sequence])[0]


def calculate_lexical_similarity(x_sequences, y_sequences, tfidf_vectorizer):
    """
    Computes lexical similarity between two lists of texts.

    lexical_similarity = cos(x_vector, y_vector)
        where x_vector and y_vector are tf-idf representations of texts.
    """
    x_sequence = ' '.join(x_sequences)
    y_sequence = ' '.join(y_sequences)
    x = _calculate_tfidf_vector(tfidf_vectorizer, x_sequence)
    y = _calculate_tfidf_vector(tfidf_vectorizer, y_sequence)

    # Compute dot-product between two 1xk-sparse matrices.
    # We also need to slice [0, 0] because the result of .dot() operation is a 1x1 sparse matrix,
    # but we want to extract a float number.
    return x.dot(y.T)[0, 0]


def _calculate_tfidf_vectorizer(base_corpus_name=BASE_CORPUS_NAME):
    index_to_token = load_index_to_item(get_index_to_token_path(base_corpus_name))
    token_to_index = {v: k for k, v in index_to_token.items()}
    train_lines = _load_train_lines()
    tfidf_vectorizer = TfidfVectorizer(tokenizer=get_tokens_sequence, vocabulary=token_to_index)
    tfidf_vectorizer.fit(train_lines)
    return tfidf_vectorizer


def get_tfidf_vectorizer():
    return get_persisted(_calculate_tfidf_vectorizer, _TFIDF_VECTORIZER_FULL_PATH)
