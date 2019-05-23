import os
import pickle
import tarfile
from functools import partial
from abc import abstractmethod, ABCMeta

from cakechat.utils.logger import get_logger, WithLogger

_logger = get_logger(__name__)

DEFAULT_CSV_DELIMITER = ','


class AbstractFileResolver(object, metaclass=ABCMeta):
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


class PackageResolver(WithLogger):
    def __init__(self, package_path, package_file_resolver_factory, package_file_ext, package_extractor):
        """
        :param package_path:
        :param package_file_resolver_factory: a factory creating package file resolver
        :param package_file_ext: package file extension
        :param package_extractor: a function taking package file, package path, and extracting contents to that path
        :return:
        """
        WithLogger.__init__(self)

        self._package_path = package_path
        self._package_file_resolver_factory = package_file_resolver_factory
        self._package_file_ext = package_file_ext
        self._package_extractor = package_extractor

    @staticmethod
    def init_resolver(**kwargs):
        """
        Method helping to set once some parameters like package_file_resolver and package_extractor

        :param kwargs:
        :return: partially initialized class object
        """
        return partial(PackageResolver, **kwargs)

    def resolve(self):
        if os.path.exists(self._package_path):
            return True

        package_file_path = '{}.{}'.format(self._package_path, self._package_file_ext)
        package_file_resolver = self._package_file_resolver_factory(package_file_path)
        if package_file_resolver.resolve():
            self._logger.info('Extracting package {}'.format(package_file_resolver.file_path))
            self._package_extractor(package_file_resolver.file_path, self._package_path)
            return True
        else:
            return False


def load_file(file_path, filter_empty_lines=True):
    with open(file_path, 'r', encoding='utf-8') as fh:
        lines = [line.strip() for line in fh.readlines()]
        if filter_empty_lines:
            lines = list(filter(None, lines))

        return lines


def ensure_dir(dir_name):
    if dir_name and not os.path.exists(dir_name):
        os.makedirs(dir_name)


def serialize(filename, data, protocol=pickle.HIGHEST_PROTOCOL):
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
    if os.path.exists(persisted_file_name):
        _logger.info('Loading {}'.format(persisted_file_name))
        cached = deserialize(persisted_file_name)
        return cached

    _logger.info('Creating {}'.format(persisted_file_name))
    data = factory()
    serialize(persisted_file_name, data, **kwargs)
    return data


def is_non_empty_file(file_path):
    return os.path.isfile(file_path) and os.stat(file_path).st_size != 0


class FileNotFoundException(Exception):
    pass


def extract_tar(source_path, destination_path, compression_type='gz'):
    """
    :param source_path:
    :param destination_path:
    :param compression_type: None, gz or bzip2
    :return:
    """
    mode = 'r:{}'.format(compression_type if compression_type else 'r')
    with tarfile.open(source_path, mode) as fh:
        fh.extractall(path=destination_path)


def ensure_file(file_name, mode, encoding=None):
    ensure_dir(os.path.dirname(file_name))
    return open(file_name, mode, encoding=encoding)


def get_cached(factory, cache_file_name, **kwargs):
    """
    Loads cache if exists, otherwise calls factory and stores the results in the specified cache file.
    **kwargs are passed to the serialize() function
    :param factory:
    :param cache_file_name:
    :return:
    """
    if os.path.exists(cache_file_name):
        _logger.info('Loading {}'.format(cache_file_name))
        cached = deserialize(cache_file_name)
        return cached

    _logger.info('Creating {}'.format(cache_file_name))
    data = factory()
    serialize(cache_file_name, data, **kwargs)
    return data
