import os
import numpy as np

from cakechat.config import W2V_MODEL_DIR, USE_SKIP_GRAM, W2V_WINDOW_SIZE, TOKEN_REPRESENTATION_SIZE
from cakechat.utils.logger import get_logger
from cakechat.utils.text_processing import SPECIAL_TOKENS

_logger = get_logger(__name__)


def _get_w2v_model_name(corpus_name, voc_size, vec_size, window_size=W2V_WINDOW_SIZE, skip_gram=USE_SKIP_GRAM):
    params_str = get_w2v_params_str(voc_size, vec_size, window_size, skip_gram)
    model_name = '{}_{}'.format(corpus_name, params_str)
    return model_name


def get_w2v_params_str(voc_size, vec_size, window_size=W2V_WINDOW_SIZE, skip_gram=USE_SKIP_GRAM):
    params_str = 'window{window_size}_voc{voc_size}_vec{vec_size}_sg{skip_gram}'
    params_str = params_str.format(window_size=window_size, voc_size=voc_size, vec_size=vec_size, skip_gram=skip_gram)
    return params_str


def get_w2v_model_path(corpus_name, voc_size, vec_size, window_size=W2V_WINDOW_SIZE, skip_gram=USE_SKIP_GRAM):
    model_name = get_w2v_model_name(corpus_name, voc_size, vec_size, window_size, skip_gram)
    model_path = os.path.join(W2V_MODEL_DIR, '{}.bin'.format(model_name))
    return model_path


def get_w2v_model_name(corpus_name, voc_size, vec_size, window_size=W2V_WINDOW_SIZE, skip_gram=USE_SKIP_GRAM):
    params_str = get_w2v_params_str(voc_size, vec_size, window_size, skip_gram)
    model_name = '{}_{}'.format(corpus_name, params_str)
    return model_name


def get_token_vector(token, model, token_vec_size=TOKEN_REPRESENTATION_SIZE):
    if token in model.wv.vocab:
        return np.array(model[token])

    # generally we want have trained embeddings for all special tokens except the PAD one
    if token != SPECIAL_TOKENS.PAD_TOKEN:
        _logger.warn('Unknown embedding for token "{}"'.format(token))

    return np.zeros(token_vec_size, dtype=np.float32)
