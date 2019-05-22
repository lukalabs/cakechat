import abc
import hashlib
import json
import os

from cakechat.dialog_model.quality.metrics.utils import MetricsSerializer
from cakechat.utils.files_utils import DummyFileResolver
from cakechat.utils.logger import WithLogger


class AbstractModel(WithLogger, metaclass=abc.ABCMeta):
    # Model resources default values
    _MODEL_RESOURCE_NAME = 'model'
    _METRICS_RESOURCE_NAME = 'metrics'

    def __init__(self, model_resolver_factory=None, metrics_serializer=None):
        """
        :param model_resolver_factory: a factory of `cakechat.utils.files_utils.AbstractFileResolver` that
            takes model path and returns a file resolver object
        :param metrics_serializer: an instance compatible with the interface of
        `cakechat.dialog_model.quality.metrics.utils.MetricsSerializer`
        :return:
        """
        super(AbstractModel, self).__init__()

        self._model = None
        self.__model_resolver_factory = model_resolver_factory if model_resolver_factory else DummyFileResolver

        self._model_init_path = None

        self._metrics = None
        self._metrics_serializer = metrics_serializer if metrics_serializer else MetricsSerializer()

    @property
    def model(self):
        return self._model

    @property
    def metrics(self):
        return self._metrics

    @property
    def _model_params_str(self):
        return json.dumps(self.model_params, sort_keys=True)

    @property
    def model_id(self):
        if not hasattr(self, '__uniq_id'):
            self.__uniq_id = hashlib.md5(self._model_params_str.encode()).hexdigest()

        # it's enough to use only the first 12 characters of the hash to avoid collisions
        # see: http://stackoverflow.com/a/18134919
        short_id = self.__uniq_id[:12]
        return '{}_{}'.format(self.model_name, short_id)

    @property
    def model_path(self):
        return os.path.join(self._model_dir, self.model_id)

    @property
    def _model_resource_path(self):
        return os.path.join(self.model_path, self._MODEL_RESOURCE_NAME)

    @property
    def _metrics_resource_path(self):
        return os.path.join(self.model_path, self._METRICS_RESOURCE_NAME)

    @property
    @abc.abstractmethod
    def model_name(self):
        """
        Returns human-readable model name

        :return:
        """
        pass

    @property
    @abc.abstractmethod
    def model_params(self):
        """
        Returns a dict with model params. Note that these params are used to compute a unique model id, so they should
        reflect the full model state (including the data (or its id) on which the model is trained/validated).

        :return:
        """
        pass

    @property
    @abc.abstractmethod
    def _model_dir(self):
        pass

    @abc.abstractmethod
    def train_model(self, *args, **kwargs):
        """
        Trains the model. Put just for method name unification

        :param args:
        :param kwargs:
        :return:
        """
        pass

    @abc.abstractmethod
    def _load_model(self, fresh_model, model_resource_path):
        """
        Fills given fresh_model by parameters stored in model_resource_path

        :param fresh_model: model object
        :param model_resource_path:
        :return: loaded model
        """
        pass

    @abc.abstractmethod
    def _save_model(self, model_resource_path):
        pass

    @abc.abstractmethod
    def _evaluate(self):
        """
        Evaluates model on validation set/sets

        :return: { dataset_name : { metric_name : metric_value } }
        """
        pass

    def resolve_model(self):
        self._logger.info('Looking for the previously trained model')
        self._logger.info('Model params str: {}'.format(self._model_params_str))
        model_path = self._model_init_path or self._model_resource_path

        if not self._model_init_path and not self.__model_resolver_factory(self.model_path).resolve():
            err_msg = 'Can\'t find previously trained model in {}'.format(self.model_path)
            self._logger.error(err_msg)
            raise ValueError(err_msg)

        self._logger.info('Loading previously calculated model')
        self._model = self._load_model(self._model, model_path)
        self._logger.info('Loaded model: {}'.format(model_path))
