import os

from cakechat.config import W2V_WINDOW_SIZE, W2V_MODEL_DIR, USE_SKIP_GRAM


def get_w2v_params_str(voc_size, vec_size, window_size=W2V_WINDOW_SIZE, skip_gram=USE_SKIP_GRAM):
    params_str = 'window{window_size}_voc{voc_size}_vec{vec_size}_sg{skip_gram}'
    params_str = params_str.format(window_size=window_size, voc_size=voc_size, vec_size=vec_size, skip_gram=skip_gram)
    return params_str


def _get_w2v_model_name(corpus_name, voc_size, vec_size, window_size=W2V_WINDOW_SIZE, skip_gram=USE_SKIP_GRAM):
    params_str = get_w2v_params_str(voc_size, vec_size, window_size, skip_gram)
    model_name = '%s_%s' % (corpus_name, params_str)
    return model_name


def get_w2v_model_path(corpus_name, voc_size, vec_size, window_size=W2V_WINDOW_SIZE, skip_gram=USE_SKIP_GRAM):
    model_name = _get_w2v_model_name(corpus_name, voc_size, vec_size, window_size, skip_gram)
    model_path = os.path.join(W2V_MODEL_DIR, '%s.bin' % model_name)
    return model_path
