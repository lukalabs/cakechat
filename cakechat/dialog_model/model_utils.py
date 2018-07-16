from itertools import islice

import numpy as np
import theano
from six import text_type
from six.moves import xrange, map, zip

from cakechat.config import TRAIN_CORPUS_NAME, WORD_EMBEDDING_DIMENSION, INPUT_CONTEXT_SIZE, INPUT_SEQUENCE_LENGTH, DEFAULT_CONDITION, \
    S3_MODELS_BUCKET_NAME, S3_W2V_REMOTE_DIR, OUTPUT_SEQUENCE_LENGTH, W2V_WINDOW_SIZE, USE_SKIP_GRAM
from cakechat.utils.data_types import Dataset
from cakechat.utils.logger import get_logger
from cakechat.utils.s3 import S3FileResolver
from cakechat.utils.tee_file import file_buffered_tee
from cakechat.utils.text_processing import SPECIAL_TOKENS
from cakechat.utils.w2v import get_w2v_model

_logger = get_logger(__name__)


def transform_conditions_to_ids(conditions, condition_to_index, n_dialogs):
    condition_ids_iterator = map(
        lambda condition: condition_to_index.get(condition, condition_to_index[DEFAULT_CONDITION]), conditions)
    condition_ids = np.full(n_dialogs, condition_to_index[DEFAULT_CONDITION], dtype=np.int32)
    for sample_idx, condition_id in enumerate(condition_ids_iterator):
        condition_ids[sample_idx] = condition_id
    return condition_ids


def lines_to_context(tokenized_lines):
    for line in tokenized_lines:
        yield [line]


def transform_contexts_to_token_ids(tokenized_contexts,
                                    token_to_index,
                                    max_line_len,
                                    max_context_len=1,
                                    max_contexts_num=None,
                                    add_start_end=False):
    """
    Transforms contexts of lines of text to matrix of indices of tokens to be used in training/predicting.
    Uses only first max_lines_num lines of tokenized_lines. Also clips each line to max_line_len tokens.
    if length of a line is less that max_line_len, it's padded with token_to_index[PAD_TOKEN].

    :param tokenized_contexts: iterable of lists (contexts) of lists (utterances) of tokens to transform to ids
    :param token_to_index: dict that maps each token to its id
    :param max_line_len: maximum number of tokens in a line
    :param max_context_len: maximum context length
    :param max_contexts_num: maximum number of contexts
    :param add_start_end: add start/end tokens to sequence
    :return: X -- numpy array, dtype=np.int32, shape = (max_lines_num, max_context_len, max_line_len).
    """

    if max_contexts_num is None:
        if not isinstance(tokenized_contexts, list):
            raise TypeError('tokenized_lines should has list type if max_lines_num is not specified')
        max_contexts_num = len(tokenized_contexts)

    X = np.full(
        (max_contexts_num, max_context_len, max_line_len), token_to_index[SPECIAL_TOKENS.PAD_TOKEN], dtype=np.int32)

    for context_idx, context in enumerate(tokenized_contexts):
        if context_idx >= max_contexts_num:
            break

        # take last max_content_len utterances
        context = context[-max_context_len:]

        # fill utterances to the end of context, keep first empty utterances padded.
        utterance_offset = max_context_len - len(context)
        for utterance_idx, utterance in enumerate(context):
            if add_start_end:
                utterance = [SPECIAL_TOKENS.START_TOKEN] + utterance + [SPECIAL_TOKENS.EOS_TOKEN]

            for token_idx, token in enumerate(utterance[:max_line_len]):
                X[context_idx, utterance_offset + utterance_idx, token_idx] = token_to_index[token] \
                    if token in token_to_index else token_to_index[SPECIAL_TOKENS.UNKNOWN_TOKEN]

    return X


def transform_lines_to_token_ids(tokenized_lines, token_to_index, max_line_len, max_lines_num=None,
                                 add_start_end=False):
    """
    Transforms lines of text to matrix of indices of tokens to be used in training/predicting.
    Uses only first max_lines_num lines of tokenized_lines. Also clips each line to max_line_len tokens.
    if length of a line is less that max_line_len, it's padded with token_to_index[PAD_TOKEN].

    :param tokenized_lines: iterable of lists (utterances) of tokens to transform to ids
    :param token_to_index: dict that maps each token to its id
    :param max_line_len: maximum number of tokens in a lineh
    :param max_lines_num: maximum number of lines
    :param add_start_end: add start/end tokens to sequence
    :return: X -- numpy array, dtype=np.int32, shape = (max_lines_num, max_line_len).
    """

    if max_lines_num is None:
        if not isinstance(tokenized_lines, list):
            raise TypeError('tokenized_lines should has list type if max_lines_num is not specified')
        max_lines_num = len(tokenized_lines)

    X = np.full((max_lines_num, max_line_len), token_to_index[SPECIAL_TOKENS.PAD_TOKEN], dtype=np.int32)

    for line_idx, line in enumerate(tokenized_lines):
        if line_idx >= max_lines_num:
            break

        if add_start_end:
            line = [SPECIAL_TOKENS.START_TOKEN] + line + [SPECIAL_TOKENS.EOS_TOKEN]

        for token_idx, token in enumerate(line[:max_line_len]):
            X[line_idx, token_idx] = token_to_index[token] \
                if token in token_to_index else token_to_index[SPECIAL_TOKENS.UNKNOWN_TOKEN]

    return X


def transform_token_ids_to_sentences(y_ids, index_to_token):
    """
    Transforms batch of token ids into list of joined strings (sentences)
    Transformation of each sentence ends when the end_token occurred.
    Skips start tokens.

    :param y_ids: numpy array of integers, shape (lines_num, tokens_num), represents token ids
    :param index_to_token: dictionary to be used for transforming
    :return: list of strings, list length = lines_num
    """
    n_responses, n_tokens = y_ids.shape

    responses = []
    for resp_idx in xrange(n_responses):
        response_tokens = []
        for token_idx in xrange(n_tokens):
            token_to_add = index_to_token[y_ids[resp_idx, token_idx]]

            if token_to_add in [SPECIAL_TOKENS.EOS_TOKEN, SPECIAL_TOKENS.PAD_TOKEN]:
                break
            if token_to_add == SPECIAL_TOKENS.START_TOKEN:
                continue
            response_tokens.append(token_to_add)

        response_str = ' '.join(response_tokens)
        if not isinstance(response_str, text_type):
            response_str = response_str.decode('utf-8')

        responses.append(response_str)
    return responses


def transform_context_token_ids_to_sentences(x_ids, index_to_token):
    """
    Transforms batch of token ids into list of joined strings (sentences)
    Transformation of each sentence ends when the end_token occurred.
    Skips start tokens.

    :param x_ids: context token ids, numpy array of shape (batch_size, context_len, tokens_num)
    :param index_to_token:
    :return:
    """
    n_samples, n_contexts, n_tokens = x_ids.shape

    samples = []
    for sample_idx in xrange(n_samples):
        context_samples = []
        for cont_idx in xrange(n_contexts):
            sample_tokens = []
            for token_idx in xrange(n_tokens):
                token_to_add = index_to_token[x_ids[sample_idx, cont_idx, token_idx]]

                if token_to_add == SPECIAL_TOKENS.EOS_TOKEN or token_to_add == SPECIAL_TOKENS.PAD_TOKEN:
                    break
                if token_to_add == SPECIAL_TOKENS.START_TOKEN:
                    continue

                sample_tokens.append(token_to_add)

            sample_str = ' '.join(sample_tokens)
            if not isinstance(sample_str, text_type):
                sample_str = sample_str.decode('utf-8')

            context_samples.append(sample_str)
        samples.append(' / '.join(context_samples))
    return samples


def _get_token_vector(token, w2v_model):
    if token in w2v_model.wv.vocab:
        return np.array(w2v_model[token])
    elif token == SPECIAL_TOKENS.PAD_TOKEN:
        return np.zeros(w2v_model.vector_size)
    else:
        _logger.warning('Can\'t find token [%s] in w2v dict' % token)
        if not hasattr(_get_token_vector, 'unk_vector'):
            if SPECIAL_TOKENS.UNKNOWN_TOKEN in w2v_model.wv.vocab:
                _get_token_vector.unk_vector = np.array(w2v_model[SPECIAL_TOKENS.UNKNOWN_TOKEN])
            else:
                _get_token_vector.unk_vector = np.mean([w2v_model[x] for x in w2v_model.wv.vocab], axis=0)
        return _get_token_vector.unk_vector


def transform_w2v_model_to_matrix(w2v_model, index_to_token):
    _logger.info('Preparing embedding matrix based on w2v_model and index_to_token dict')

    token_to_index = {v: k for k, v in index_to_token.items()}
    tokens_num = len(index_to_token)
    output = np.zeros((tokens_num, WORD_EMBEDDING_DIMENSION), dtype=theano.config.floatX)
    for token in index_to_token.values():
        idx = token_to_index[token]
        output[idx] = _get_token_vector(token, w2v_model)

    return output


def get_w2v_embedding_matrix(tokenized_dialog_lines, index_to_token, add_start_end=False):
    if add_start_end:
        tokenized_dialog_lines = (
            [SPECIAL_TOKENS.START_TOKEN] + line + [SPECIAL_TOKENS.EOS_TOKEN] for line in tokenized_dialog_lines)

    w2v_resolver_factory = S3FileResolver.init_resolver(bucket_name=S3_MODELS_BUCKET_NAME, remote_dir=S3_W2V_REMOTE_DIR)

    w2v_model = get_w2v_model(
        TRAIN_CORPUS_NAME,
        len(index_to_token),
        model_resolver_factory=w2v_resolver_factory,
        tokenized_lines=tokenized_dialog_lines,
        vec_size=WORD_EMBEDDING_DIMENSION,
        window_size=W2V_WINDOW_SIZE,
        skip_gram=USE_SKIP_GRAM)
    w2v_matrix = transform_w2v_model_to_matrix(w2v_model, index_to_token)
    return w2v_matrix


def get_training_batch(inputs, batch_size, random_permute=False):
    n_samples = inputs[0].shape[0]
    n_batches = n_samples // batch_size
    batches_seq = np.arange(n_batches)
    samples_seq = np.arange(n_samples)

    if random_permute:
        np.random.shuffle(samples_seq)

    for i in batches_seq:
        start = i * batch_size
        end = (i + 1) * batch_size
        samples_ids = samples_seq[start:end]
        # yield batch_size selected sequences of x and y ids
        yield tuple(inp[samples_ids] for inp in inputs)

    seen_samples_num = len(batches_seq) * batch_size

    if seen_samples_num < n_samples:
        samples_ids = samples_seq[seen_samples_num:]
        # yield the rest of x and y sequences
        yield tuple(inp[samples_ids] for inp in inputs)


def reverse_nn_input(dataset, service_tokens):
    """
    Swaps the last utterance of x with y for each x-y pair in the dataset.
    To handle different length of sequences, everything is filled with pads
    to the length of longest sequence.
    """
    # Swap last utterance of x with y, while padding with start- and eos-tokens
    y_output = np.full(dataset.y.shape, service_tokens.pad_token_id, dtype=dataset.y.dtype)
    for y_output_sample, x_input_sample in zip(y_output, dataset.x[:, -1]):
        # Write start token at the first index
        y_output_sample[0] = service_tokens.start_token_id
        y_output_token_index = 1
        for value in x_input_sample:
            # We should stop at pad tokens in the input sample
            if value == service_tokens.pad_token_id:
                break
            # We should keep last token index with pad, so we can replace it futher with eos-token
            if y_output_token_index == y_output_sample.shape[-1] - 1:
                break
            y_output_sample[y_output_token_index] = value
            y_output_token_index += 1
        # Write eos token right after the last non-pad token in the sample
        y_output_sample[y_output_token_index] = service_tokens.eos_token_id

    # Use utterances from y in x while truncating start- and eos-tokens
    x_output = np.full(dataset.x.shape, service_tokens.pad_token_id, dtype=dataset.x.dtype)
    for x_output_sample, x_input_sample, y_input_sample in zip(x_output, dataset.x[:, :-1], dataset.y):
        # Copy all the context utterances except the last one right to the output
        x_output_sample[:-1] = x_input_sample
        x_output_token_index = 0
        for value in y_input_sample:
            # Skip start- and eos-tokens from the input sample because we don't need them in X
            if value in {service_tokens.start_token_id, service_tokens.eos_token_id}:
                continue
            # Stop if we already reached the end of output sample (in case the input sample is longer than output)
            if x_output_token_index == x_output_sample.shape[-1]:
                break
            # Fill the tokens of the last utterance in dialog context
            x_output_sample[-1, x_output_token_index] = value
            x_output_token_index += 1

    return Dataset(x=x_output, y=y_output, condition_ids=dataset.condition_ids)


def transform_conditions_to_nn_input(dialog_conditions, condition_to_index, num_dialogs):
    y_conditions_iterator = islice(dialog_conditions, 1, None, 2)

    _logger.info('Iterating through conditions of output list')
    return transform_conditions_to_ids(y_conditions_iterator, condition_to_index, num_dialogs)


def _get_x_data_iterator_with_context(x_data_iterator, y_data_iterator, context_size=INPUT_CONTEXT_SIZE):
    context = []

    last_y_line = None
    for x_line, y_line in zip(x_data_iterator, y_data_iterator):
        if x_line != last_y_line:
            context = []  # clear context if last response != current dialog context (new dialog)

        context.append(x_line)
        yield context[-context_size:]  # yield list of tokenized lines
        last_y_line = y_line


def transform_lines_to_nn_input(tokenized_dialog_lines, token_to_index):
    """
    Splits lines (IterableSentences) and generates numpy arrays of token ids suitable for training.
    Doesn't store all lines in memory.
    """
    x_data_iterator, y_data_iterator, iterator_for_len_calc = file_buffered_tee(tokenized_dialog_lines, 3)

    _logger.info('Iterating through lines to get number of elements in the dataset')
    n_dialogs = sum(1 for _ in iterator_for_len_calc)

    x_data_iterator = islice(x_data_iterator, 0, None, 2)
    y_data_iterator = islice(y_data_iterator, 1, None, 2)
    n_dialogs //= 2

    y_data_iterator, y_data_iterator_for_context = file_buffered_tee(y_data_iterator)
    x_data_iterator = _get_x_data_iterator_with_context(x_data_iterator, y_data_iterator_for_context)

    _logger.info('Iterating through lines to get input matrix')
    x_ids = transform_contexts_to_token_ids(
        x_data_iterator, token_to_index, INPUT_SEQUENCE_LENGTH, INPUT_CONTEXT_SIZE, max_contexts_num=n_dialogs)

    _logger.info('Iterating through lines to get output matrix')
    y_ids = transform_lines_to_token_ids(
        y_data_iterator, token_to_index, OUTPUT_SEQUENCE_LENGTH, n_dialogs, add_start_end=True)
    return x_ids, y_ids, n_dialogs
