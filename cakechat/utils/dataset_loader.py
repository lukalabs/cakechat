import os
from itertools import islice

import numpy as np

from cakechat.config import TEST_DATA_DIR, TRAIN_CORPUS_NAME, CONTEXT_FREE_VAL_CORPUS_NAME, QUESTIONS_CORPUS_NAME, \
    CONTEXT_SENSITIVE_VAL_CORPUS_NAME, INPUT_SEQUENCE_LENGTH, INPUT_CONTEXT_SIZE, TRAIN_SUBSET_SIZE, \
    DEFAULT_CONDITION, RANDOM_SEED, MAX_VAL_LINES_NUM
from cakechat.dialog_model.model_utils import lines_to_context, transform_contexts_to_token_ids, \
    transform_conditions_to_nn_input, transform_lines_to_nn_input
from cakechat.utils.data_types import Dataset
from cakechat.utils.files_utils import load_file, is_non_empty_file
from cakechat.utils.logger import get_logger
from cakechat.utils.text_processing import get_tokens_sequence, replace_out_of_voc_tokens, \
    get_processed_corpus_path, load_processed_dialogs_from_json, FileTextLinesIterator, \
    get_dialog_lines_and_conditions, ProcessedLinesIterator, get_alternated_dialogs_lines

_logger = get_logger(__name__)


def get_tokenized_test_lines(corpus_name, tokens_voc):
    corpus_path = os.path.join(TEST_DATA_DIR, '%s.txt' % corpus_name)
    if not is_non_empty_file(corpus_path):
        raise ValueError('Test corpus file doesn\'t exist: {}'.format(corpus_path))
    test_lines = load_file(corpus_path)
    result = []
    for line in test_lines:
        tokenized_line = get_tokens_sequence(line)
        tokenized_line = replace_out_of_voc_tokens(tokenized_line, tokens_voc)
        result.append(tokenized_line)

    return result


def _load_dataset_without_responses(corpus_name, token_to_index):
    tokenized_lines = get_tokenized_test_lines(corpus_name, set(token_to_index.keys()))
    context_tokens_ids = transform_contexts_to_token_ids(
        lines_to_context(tokenized_lines),
        token_to_index,
        INPUT_SEQUENCE_LENGTH,
        INPUT_CONTEXT_SIZE,
        max_contexts_num=len(tokenized_lines))
    return Dataset(x=context_tokens_ids, y=None, condition_ids=None)


def load_questions_set(token_to_index):
    return _load_dataset_without_responses(QUESTIONS_CORPUS_NAME, token_to_index)


def load_context_free_val(token_to_index):
    _logger.info('Transform context free validation lines to matrix of indexes')
    tokenized_validation_lines = get_tokenized_test_lines(CONTEXT_FREE_VAL_CORPUS_NAME, set(token_to_index.keys()))
    tokenized_validation_lines = tokenized_validation_lines[:MAX_VAL_LINES_NUM]
    x_validation, y_validation, _ = transform_lines_to_nn_input(tokenized_validation_lines, token_to_index)
    return Dataset(x=x_validation, y=y_validation, condition_ids=None)


def load_context_sensitive_val(token_to_index, condition_to_index):
    processed_val_corpus_path = get_processed_corpus_path(CONTEXT_SENSITIVE_VAL_CORPUS_NAME)
    context_sensitive_val_dialogs = load_processed_dialogs_from_json(
        FileTextLinesIterator(processed_val_corpus_path), text_field_name='text', condition_field_name='condition')
    context_sensitive_val_dialogs = islice(context_sensitive_val_dialogs, MAX_VAL_LINES_NUM)

    alternated_context_sensitive_val_dialogs = \
        get_alternated_dialogs_lines(context_sensitive_val_dialogs)
    alternated_context_sensitive_val_lines, alternated_context_sensitive_val_conditions = \
        get_dialog_lines_and_conditions(alternated_context_sensitive_val_dialogs,
                                        text_field_name='text', condition_field_name='condition')
    tokenized_alternated_context_sensitive_val_lines = ProcessedLinesIterator(
        alternated_context_sensitive_val_lines, processing_callbacks=[get_tokens_sequence])

    _logger.info('Transform context sensitive validation lines to tensor of indexes')
    x_context_sensitive_val, y_context_sensitive_val, num_context_sensitive_val_dialogs = \
        transform_lines_to_nn_input(tokenized_alternated_context_sensitive_val_lines, token_to_index)
    condition_ids_context_sensitive_val = transform_conditions_to_nn_input(
        alternated_context_sensitive_val_conditions, condition_to_index, num_context_sensitive_val_dialogs)
    return Dataset(
        x=x_context_sensitive_val, y=y_context_sensitive_val, condition_ids=condition_ids_context_sensitive_val)


def load_conditioned_train_set(token_to_index, condition_to_index, train_subset_size=TRAIN_SUBSET_SIZE):
    processed_corpus_path = get_processed_corpus_path(TRAIN_CORPUS_NAME)
    dialogs = load_processed_dialogs_from_json(
        FileTextLinesIterator(processed_corpus_path), text_field_name='text', condition_field_name='condition')
    if train_subset_size:
        dialogs = islice(dialogs, train_subset_size)
    train_lines, train_conditions = get_dialog_lines_and_conditions(
        get_alternated_dialogs_lines(dialogs), text_field_name='text', condition_field_name='condition')
    tokenized_alternated_train_lines = ProcessedLinesIterator(train_lines, processing_callbacks=[get_tokens_sequence])

    # prepare train set
    x_train, y_train, n_dialogs = transform_lines_to_nn_input(tokenized_alternated_train_lines, token_to_index)

    condition_ids_train = transform_conditions_to_nn_input(train_conditions, condition_to_index, n_dialogs)
    return Dataset(x=x_train, y=y_train, condition_ids=condition_ids_train)


def generate_subset(dataset, subset_size, random_seed=RANDOM_SEED):
    # Fix random seed here so that we get the same subsets every time the function is called
    np.random.seed(random_seed)
    if subset_size > dataset.x.shape[0]:
        raise ValueError('Error while generating subset of the validation data: '
                         'dataset size ({}) is less than subset size ({})'.format(dataset.x.shape[0], subset_size))
    sample_idx = np.random.choice(dataset.x.shape[0], size=subset_size, replace=False)
    return Dataset(
        x=dataset.x[sample_idx],
        y=dataset.y[sample_idx] if dataset.y is not None else None,
        condition_ids=dataset.condition_ids[sample_idx] if dataset.condition_ids is not None else None)


def load_datasets(token_to_index, condition_to_index):
    train = load_conditioned_train_set(token_to_index, condition_to_index)
    validation = load_context_free_val(token_to_index)
    questions = load_questions_set(token_to_index)

    validation_set_size = validation.x.shape[0]

    train_subset = generate_subset(train, validation_set_size)

    # prepare conditioned subset
    defined_condition_mask = train.condition_ids != condition_to_index[DEFAULT_CONDITION]
    defined_condition_dataset = Dataset(
        x=train.x[defined_condition_mask],
        y=train.y[defined_condition_mask],
        condition_ids=train.condition_ids[defined_condition_mask])

    defined_condition_dataset_len = defined_condition_dataset.x.shape[0]
    defined_condition_subset = generate_subset(defined_condition_dataset,
                                               min(validation_set_size, defined_condition_dataset_len))

    return train, questions, validation, train_subset, defined_condition_subset
