import os

from cachetools import cached

from cakechat.config import BASE_CORPUS_NAME, S3_MODELS_BUCKET_NAME, S3_TOKENS_IDX_REMOTE_DIR, \
    S3_NN_MODEL_REMOTE_DIR, S3_CONDITIONS_IDX_REMOTE_DIR, PREDICTION_MODES, TRAIN_CORPUS_NAME, \
    USE_PRETRAINED_W2V_EMBEDDINGS_LAYER
from cakechat.dialog_model.inference_model import InferenceCakeChatModel
from cakechat.utils.data_types import ModelParam
from cakechat.utils.dataset_loader import get_validation_data_id, get_validation_sets_names
from cakechat.utils.files_utils import FileNotFoundException
from cakechat.utils.s3 import S3FileResolver, get_s3_model_resolver
from cakechat.utils.text_processing import get_index_to_token_path, load_index_to_item, get_index_to_condition_path
from cakechat.utils.w2v.model import get_w2v_model_id


def _get_index_to_token(fetch_from_s3):
    index_to_token_path = get_index_to_token_path(BASE_CORPUS_NAME)
    file_name = os.path.basename(index_to_token_path)
    if fetch_from_s3:
        tokens_idx_resolver = S3FileResolver(index_to_token_path, S3_MODELS_BUCKET_NAME, S3_TOKENS_IDX_REMOTE_DIR)
        if not tokens_idx_resolver.resolve():
            raise FileNotFoundException('No such file on S3: {}'.format(file_name))
    else:
        if not os.path.exists(index_to_token_path):
            raise FileNotFoundException('No such file: {}'.format(file_name) +
                                        'Run "python tools/fetch.py" first to get all necessary files.')

    return load_index_to_item(index_to_token_path)


def _get_index_to_condition(fetch_from_s3):
    index_to_condition_path = get_index_to_condition_path(BASE_CORPUS_NAME)
    if fetch_from_s3:
        index_to_condition_resolver = S3FileResolver(index_to_condition_path, S3_MODELS_BUCKET_NAME,
                                                     S3_CONDITIONS_IDX_REMOTE_DIR)
        if not index_to_condition_resolver.resolve():
            raise FileNotFoundException('Can\'t get index_to_condition because file does not exist on S3')
    else:
        if not os.path.exists(index_to_condition_path):
            raise FileNotFoundException('Can\'t get index_to_condition because file does not exist. '
                                        'Run tools/fetch.py first to get all required files or construct '
                                        'it yourself.')

    return load_index_to_item(index_to_condition_path)


@cached(cache={})
def get_trained_model(is_reverse_model=False, reverse_model=None, fetch_from_s3=True):
    """
    Get a trained model, direct or reverse.
    :param is_reverse_model: boolean, if True, a reverse trained model will be returned to be used during inference
    in direct model in *_reranking modes; if False, a direct trained model is returned
    :param reverse_model: object of a reverse model to be used in direct model in *_reranking inference modes
    :param fetch_from_s3: boolean, if True, download trained model from Amazon S3 if the the model is not found locally;
    if False, the model presence will be checked only locally
    :return:
    """
    if fetch_from_s3:
        resolver_factory = get_s3_model_resolver(S3_MODELS_BUCKET_NAME, S3_NN_MODEL_REMOTE_DIR)
    else:
        resolver_factory = None

    w2v_model_id = get_w2v_model_id() if USE_PRETRAINED_W2V_EMBEDDINGS_LAYER else None

    model = InferenceCakeChatModel(
        index_to_token=_get_index_to_token(fetch_from_s3),
        index_to_condition=_get_index_to_condition(fetch_from_s3),
        training_data_param=ModelParam(value=None, id=TRAIN_CORPUS_NAME),
        validation_data_param=ModelParam(value=None, id=get_validation_data_id(get_validation_sets_names())),
        w2v_model_param=ModelParam(value=None, id=w2v_model_id),
        model_resolver=resolver_factory,
        is_reverse_model=is_reverse_model,
        reverse_model=reverse_model)

    model.init_model()
    model.resolve_model()
    return model


def get_reverse_model(prediction_mode):
    reranking_modes = [PREDICTION_MODES.beamsearch_reranking, PREDICTION_MODES.sampling_reranking]
    return get_trained_model(is_reverse_model=True) if prediction_mode in reranking_modes else None
