import multiprocessing
import os

from gensim.models import Word2Vec

from cakechat.config import WORD_EMBEDDING_DIMENSION, W2V_WINDOW_SIZE, MIN_WORD_FREQ, USE_SKIP_GRAM
from cakechat.utils.files_utils import DummyFileResolver, ensure_dir
from cakechat.utils.logger import get_logger
from cakechat.utils.tee_file import file_buffered_tee
from cakechat.utils.w2v.utils import get_w2v_params_str, get_w2v_model_path

_WORKERS_NUM = multiprocessing.cpu_count()

_logger = get_logger(__name__)


def _train_model(tokenized_lines, voc_size, vec_size, window_size, skip_gram):
    _logger.info('Word2Vec model will be trained now. It can take long, so relax and have fun.')

    params_str = get_w2v_params_str(voc_size, vec_size, window_size, skip_gram)
    _logger.info('Parameters for training: %s' % params_str)

    model = Word2Vec(
        window=window_size,
        size=vec_size,
        max_vocab_size=voc_size,
        min_count=MIN_WORD_FREQ,
        workers=_WORKERS_NUM,
        sg=skip_gram)

    tokenized_lines_for_voc, tokenized_lines_for_train = file_buffered_tee(tokenized_lines)

    model.build_vocab(tokenized_lines_for_voc)
    model.train(tokenized_lines_for_train)

    # forget the original vectors and only keep the normalized ones = saves lots of memory
    # https://radimrehurek.com/gensim/models/word2vec.html#gensim.models.word2vec.Word2Vec.init_sims
    model.init_sims(replace=True)

    return model


def _save_model(model, model_path):
    _logger.info('Saving model to %s' % model_path)
    ensure_dir(os.path.dirname(model_path))
    model.save(model_path, separately=[])
    _logger.info('Model has been saved')


def _load_model(model_path):
    _logger.info('Loading model from %s' % model_path)
    model = Word2Vec.load(model_path, mmap='r')
    _logger.info('Model "%s" has been loaded.' % os.path.basename(model_path))
    return model


def get_w2v_model(corpus_name,
                  voc_size,
                  model_resolver_factory=None,
                  tokenized_lines=None,
                  vec_size=WORD_EMBEDDING_DIMENSION,
                  window_size=W2V_WINDOW_SIZE,
                  skip_gram=USE_SKIP_GRAM):
    _logger.info('Getting w2v model')

    model_path = get_w2v_model_path(corpus_name, voc_size, vec_size, window_size, skip_gram)
    model_resolver = model_resolver_factory(model_path) if model_resolver_factory else DummyFileResolver(model_path)

    if not model_resolver.resolve():
        if not tokenized_lines:
            raise ValueError('Tokenized corpus \'%s\' was not provided, so w2v model can\'t be trained.' % model_path)

        # bin model is not present on the disk, so get it
        model = _train_model(tokenized_lines, voc_size, vec_size, window_size, skip_gram)
        _save_model(model, model_path)
    else:
        # bin model is on the disk, load it
        model = _load_model(model_path)

    _logger.info('Successfully got w2v model\n')

    return model
