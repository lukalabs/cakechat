import os
import codecs
from abc import abstractmethod, ABCMeta

from six.moves import cPickle as pickle

from cakechat.utils.logger import get_logger

_logger = get_logger(__name__)


class AbstractFileResolver(object):
    __metaclass__ = ABCMeta

    def __init__(self, file_path):
        self._file_path = file_path

    @property
    def file_path(self):
        return self._file_path

    def resolve(self):
        """
        :return: True if file can be resolved, False otherwise
        """
        if os.path.exists(self._file_path):
            return True

        return self._resolve()

    @abstractmethod
    def _resolve(self):
        """
        Performs some actions if file does not exist locally. Should be defined in subclasses

        :return: True if file can be resolved, False otherwise
        """
        pass


class DummyFileResolver(AbstractFileResolver):
    """
    Does nothing if file does not exist locally
    """

    def _resolve(self):
        return False


def load_file(file_path, filter_empty_lines=True):
    with codecs.open(file_path, 'r', 'utf-8') as fh:
        lines = [line.strip() for line in fh.readlines()]
        if filter_empty_lines:
            lines = list(filter(None, lines))

        return lines


def ensure_dir(dir_name):
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)


def serialize(filename, data, protocol=2):
    ensure_dir(os.path.dirname(filename))
    with open(filename, 'wb') as f:
        pickle.dump(data, f, protocol)


def deserialize(filename):
    with open(filename, 'rb') as f:
        item = pickle.load(f)
    return item


def get_persisted(factory, persisted_file_name, **kwargs):
    """
    Loads cache if exists, otherwise calls factory and stores the results in the specified cache file.
    **kwargs are passed to the serialize() function
    :param factory:
    :param persisted_file_name:
    :return:
    """
    filename = persisted_file_name.encode('utf-8')

    if os.path.exists(filename):
        _logger.info(u'Loading {}'.format(persisted_file_name))
        cached = deserialize(filename)
        return cached

    _logger.info(u'Creating {}'.format(persisted_file_name))
    data = factory()
    serialize(filename, data, **kwargs)
    return data


def is_non_empty_file(file_path):
    return os.path.isfile(file_path) and os.stat(file_path).st_size != 0


class FileNotFoundException(Exception):
    pass
