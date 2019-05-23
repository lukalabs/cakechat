import argparse
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
import numpy
import tensorflow as tf

from cakechat.utils.env import init_keras, try_import_horovod

hvd = try_import_horovod()
init_keras(hvd)

# fix random seeds for experiments reproducibility
random.seed(42)
numpy.random.seed(42)
tf.set_random_seed(42)

from cakechat.config import BASE_CORPUS_NAME, TRAIN_CORPUS_NAME, CONTEXT_SENSITIVE_VAL_CORPUS_NAME, \
    USE_PRETRAINED_W2V_EMBEDDINGS_LAYER, S3_MODELS_BUCKET_NAME, S3_NN_MODEL_REMOTE_DIR, PREDICTION_MODE_FOR_TESTS
from cakechat.dialog_model.factory import get_reverse_model
from cakechat.dialog_model.model import CakeChatModel
from cakechat.utils.data_types import ModelParam
from cakechat.utils.dataset_loader import get_validation_data_id, get_validation_sets_names, \
    get_validation_dataset_name_to_data, get_training_dataset
from cakechat.utils.files_utils import is_non_empty_file, FileNotFoundException
from cakechat.utils.logger import get_tools_logger
from cakechat.utils.s3 import S3FileResolver
from cakechat.utils.text_processing import get_processed_corpus_path, get_index_to_token_path, \
    get_index_to_condition_path, load_index_to_item
from cakechat.utils.w2v.model import get_w2v_model_id, get_w2v_model

_logger = get_tools_logger(__file__)


def _look_for_saved_files(files_paths):
    for f_path in files_paths:
        if not is_non_empty_file(f_path):
            raise FileNotFoundException('\nCould not find the following file or it\'s empty: {0}'.format(f_path))


def train(model_init_path=None,
          is_reverse_model=False,
          train_subset_size=None,
          use_pretrained_w2v=USE_PRETRAINED_W2V_EMBEDDINGS_LAYER,
          train_corpus_name=TRAIN_CORPUS_NAME,
          context_sensitive_val_corpus_name=CONTEXT_SENSITIVE_VAL_CORPUS_NAME,
          base_corpus_name=BASE_CORPUS_NAME,
          s3_models_bucket_name=S3_MODELS_BUCKET_NAME,
          s3_nn_model_remote_dir=S3_NN_MODEL_REMOTE_DIR,
          prediction_mode_for_tests=PREDICTION_MODE_FOR_TESTS):
    processed_train_corpus_path = get_processed_corpus_path(train_corpus_name)
    processed_val_corpus_path = get_processed_corpus_path(context_sensitive_val_corpus_name)
    index_to_token_path = get_index_to_token_path(base_corpus_name)
    index_to_condition_path = get_index_to_condition_path(base_corpus_name)

    # check the existence of all necessary files before compiling the model
    _look_for_saved_files(files_paths=[processed_train_corpus_path, processed_val_corpus_path, index_to_token_path])

    # load essentials for building model and training
    index_to_token = load_index_to_item(index_to_token_path)
    index_to_condition = load_index_to_item(index_to_condition_path)
    token_to_index = {v: k for k, v in index_to_token.items()}
    condition_to_index = {v: k for k, v in index_to_condition.items()}

    training_data_param = ModelParam(
        value=get_training_dataset(train_corpus_name, token_to_index, condition_to_index, is_reverse_model,
                                   train_subset_size),
        id=train_corpus_name)

    val_sets_names = get_validation_sets_names()
    validation_data_param = ModelParam(
        value=get_validation_dataset_name_to_data(val_sets_names, token_to_index, condition_to_index, is_reverse_model),
        id=get_validation_data_id(val_sets_names))

    w2v_model_param = ModelParam(value=get_w2v_model(), id=get_w2v_model_id()) if use_pretrained_w2v \
        else ModelParam(value=None, id=None)

    model_resolver_factory = S3FileResolver.init_resolver(
        bucket_name=s3_models_bucket_name, remote_dir=s3_nn_model_remote_dir)

    reverse_model = get_reverse_model(prediction_mode_for_tests) if not is_reverse_model else None

    # build CakeChatModel
    cakechat_model = CakeChatModel(
        index_to_token,
        index_to_condition,
        training_data_param=training_data_param,
        validation_data_param=validation_data_param,
        w2v_model_param=w2v_model_param,
        model_init_path=model_init_path,
        model_resolver=model_resolver_factory,
        is_reverse_model=is_reverse_model,
        reverse_model=reverse_model,
        horovod=hvd)

    # train model
    cakechat_model.train_model()


def parse_args():
    argparser = argparse.ArgumentParser()

    argparser.add_argument(
        '-r', '--reverse', action='store_true', help='Pass this flag if you want to train reverse model.')
    argparser.add_argument(
        '-i',
        '--init_weights',
        help='Path to the file with weights that should be used for the model\'s initialisation')
    argparser.add_argument('-s', '--train-subset-size', action='store', type=int)

    return argparser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train(model_init_path=args.init_weights, is_reverse_model=args.reverse, train_subset_size=args.train_subset_size)
