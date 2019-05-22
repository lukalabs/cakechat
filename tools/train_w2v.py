import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cakechat.utils.text_processing import get_processed_corpus_path, load_processed_dialogs_from_json, \
    FileTextLinesIterator, get_dialog_lines_and_conditions, ProcessedLinesIterator, get_flatten_dialogs
from cakechat.utils.w2v.model import _get_w2v_model as get_w2v_model
from cakechat.config import TRAIN_CORPUS_NAME, VOCABULARY_MAX_SIZE, WORD_EMBEDDING_DIMENSION, W2V_WINDOW_SIZE, \
    USE_SKIP_GRAM

if __name__ == '__main__':
    processed_corpus_path = get_processed_corpus_path(TRAIN_CORPUS_NAME)

    dialogs = load_processed_dialogs_from_json(
        FileTextLinesIterator(processed_corpus_path), text_field_name='text', condition_field_name='condition')

    training_dialogs_lines_for_w2v, _ = get_dialog_lines_and_conditions(
        get_flatten_dialogs(dialogs), text_field_name='text', condition_field_name='condition')

    tokenized_training_lines = ProcessedLinesIterator(training_dialogs_lines_for_w2v, processing_callbacks=[str.split])

    get_w2v_model(
        tokenized_lines=tokenized_training_lines,
        corpus_name=TRAIN_CORPUS_NAME,
        voc_size=VOCABULARY_MAX_SIZE,
        vec_size=WORD_EMBEDDING_DIMENSION,
        window_size=W2V_WINDOW_SIZE,
        skip_gram=USE_SKIP_GRAM)
