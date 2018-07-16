import os

from cachetools import cached

from cakechat.config import BASE_CORPUS_NAME, S3_MODELS_BUCKET_NAME, S3_TOKENS_IDX_REMOTE_DIR, \
    S3_NN_MODEL_REMOTE_DIR, S3_CONDITIONS_IDX_REMOTE_DIR
from cakechat.dialog_model.model import get_nn_model
from cakechat.utils.s3 import S3FileResolver
from cakechat.utils.files_utils import FileNotFoundException
from cakechat.utils.text_processing import get_index_to_token_path, load_index_to_item, get_index_to_condition_path


def _get_index_to_token(fetch_from_s3):
    index_to_token_path = get_index_to_token_path(BASE_CORPUS_NAME)
    if fetch_from_s3:
        tokens_idx_resolver = S3FileResolver(index_to_token_path, S3_MODELS_BUCKET_NAME, S3_TOKENS_IDX_REMOTE_DIR)
        if not tokens_idx_resolver.resolve():
            raise FileNotFoundException('Can\'t get index_to_token because file does not exist at S3')
    else:
        if not os.path.exists(index_to_token_path):
            raise FileNotFoundException('Can\'t get index_to_token because file does not exist. '
                            'Run tools/download_model.py first to get all required files or construct it by yourself.')

    return load_index_to_item(index_to_token_path)


def _get_index_to_condition(fetch_from_s3):
    index_to_condition_path = get_index_to_condition_path(BASE_CORPUS_NAME)
    if fetch_from_s3:
        index_to_condition_resolver = S3FileResolver(index_to_condition_path, S3_MODELS_BUCKET_NAME,
                                                     S3_CONDITIONS_IDX_REMOTE_DIR)
        if not index_to_condition_resolver.resolve():
            raise FileNotFoundException('Can\'t get index_to_condition because file does not exist at S3')
    else:
        if not os.path.exists(index_to_condition_path):
            raise FileNotFoundException('Can\'t get index_to_condition because file does not exist. '
                            'Run tools/download_model.py first to get all required files or construct it by yourself.')

    return load_index_to_item(index_to_condition_path)


@cached(cache={})
def get_trained_model(reverse=False, fetch_from_s3=True):
    if fetch_from_s3:
        resolver_factory = S3FileResolver.init_resolver(
            bucket_name=S3_MODELS_BUCKET_NAME, remote_dir=S3_NN_MODEL_REMOTE_DIR)
    else:
        resolver_factory = None

    nn_model, model_exists = get_nn_model(index_to_token=_get_index_to_token(fetch_from_s3),
                                          index_to_condition=_get_index_to_condition(fetch_from_s3),
                                          resolver_factory=resolver_factory,
                                          is_reverse_model=reverse)
    if not model_exists:
        raise FileNotFoundException('Can\'t get the pre-trained model. Run tools/download_model.py first '
                                    'to get all required files or train it by yourself.')
    return nn_model
