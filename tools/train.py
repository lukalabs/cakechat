import argparse
import os
import sys
from six.moves import map

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cakechat.utils.env import init_theano_env

init_theano_env()

from cakechat.utils.text_processing import get_processed_corpus_path, get_index_to_token_path, \
    get_index_to_condition_path, load_processed_dialogs_from_json, load_index_to_item, FileTextLinesIterator, \
    get_flatten_dialogs, ProcessedLinesIterator, get_tokens_sequence
from cakechat.utils.files_utils import is_non_empty_file
from cakechat.utils.logger import get_tools_logger
from cakechat.utils.s3 import S3FileResolver
from cakechat.dialog_model.train import train_model
from cakechat.dialog_model.model_utils import get_w2v_embedding_matrix
from cakechat.dialog_model.model import get_nn_model
from cakechat.config import BASE_CORPUS_NAME, TRAIN_CORPUS_NAME, CONTEXT_SENSITIVE_VAL_CORPUS_NAME, \
    USE_PRETRAINED_W2V_EMBEDDINGS_LAYER, S3_MODELS_BUCKET_NAME, S3_NN_MODEL_REMOTE_DIR

_logger = get_tools_logger(__file__)


def _look_for_saved_model(nn_model_path):
    if os.path.isfile(nn_model_path):
        _logger.info('\nSaved model is found:\n{}\n'.format(nn_model_path))
    else:
        _logger.info('\nCould not find previously saved model: \n{}\nWill train it from scratch\n'.format(nn_model_path))


def _look_for_saved_files(files_paths):
    for f_path in files_paths:
        if not is_non_empty_file(f_path):
            raise Exception('\nCould not find the following file or it\'s empty: {0}'.format(f_path))


def _get_w2v_embedding_matrix_by_corpus_path(processed_train_corpus_path, index_to_token):
    if USE_PRETRAINED_W2V_EMBEDDINGS_LAYER:
        _logger.info('Getting train iterator for w2v...')
        dialogs_for_w2v = load_processed_dialogs_from_json(
            FileTextLinesIterator(processed_train_corpus_path),
            text_field_name='text',
            condition_field_name='condition')

        _logger.info('Getting text-filtered train iterator...')
        train_lines_for_w2v = map(lambda x: x['text'], get_flatten_dialogs(dialogs_for_w2v))

        _logger.info('Getting tokenized train iterator...')
        tokenized_train_lines_for_w2v = ProcessedLinesIterator(
            train_lines_for_w2v, processing_callbacks=[get_tokens_sequence])

        return get_w2v_embedding_matrix(tokenized_train_lines_for_w2v, index_to_token, add_start_end=True)
    else:
        return None


def train(init_path=None, is_reverse_model=False):
    processed_train_corpus_path = get_processed_corpus_path(TRAIN_CORPUS_NAME)
    processed_val_corpus_path = get_processed_corpus_path(CONTEXT_SENSITIVE_VAL_CORPUS_NAME)
    index_to_token_path = get_index_to_token_path(BASE_CORPUS_NAME)
    index_to_condition_path = get_index_to_condition_path(BASE_CORPUS_NAME)

    # check the existence of all necessary files before compiling the model
    _look_for_saved_files(files_paths=[processed_train_corpus_path, processed_val_corpus_path, index_to_token_path])

    index_to_token = load_index_to_item(index_to_token_path)
    index_to_condition = load_index_to_item(index_to_condition_path)

    w2v_matrix = _get_w2v_embedding_matrix_by_corpus_path(processed_train_corpus_path, index_to_token)

    # get nn_model and train it
    nn_model_resolver_factory = S3FileResolver.init_resolver(bucket_name=S3_MODELS_BUCKET_NAME,
                                                             remote_dir=S3_NN_MODEL_REMOTE_DIR)

    nn_model, _ = get_nn_model(index_to_token, index_to_condition, model_init_path=init_path, w2v_matrix=w2v_matrix,
                               resolver_factory=nn_model_resolver_factory, is_reverse_model=is_reverse_model)

    train_model(nn_model)


def parse_args():
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        '-r',
        '--reverse',
        action='store_true',
        help='Pass this flag if you want to train reverse model.')

    argparser.add_argument(
        '-i',
        '--init_weights',
        help="Path to the file with weights that should be used for the model's initialisation")

    return argparser.parse_args()


if __name__ == '__main__':
    _logger.info('THEANO_FLAGS: {}'.format(os.environ['THEANO_FLAGS']))
    if 'SLICE_TRAINSET' in os.environ:
        _logger.info('Slicing trainset to the first %d entries for faster training' % int(os.environ['SLICE_TRAINSET']))

    args = parse_args()
    train(init_path=args.init_weights, is_reverse_model=args.reverse)
