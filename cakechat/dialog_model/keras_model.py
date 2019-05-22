import abc
import os
import time
from datetime import timedelta

from keras import backend as K


from cakechat.dialog_model.abstract_callbacks import AbstractKerasModelCallback, ParametrizedCallback, \
    _KerasCallbackAdapter
from cakechat.dialog_model.abstract_model import AbstractModel
from cakechat.dialog_model.quality.metrics.plotters import DummyMetricsPlotter
from cakechat.utils.env import is_main_horovod_worker, set_horovod_worker_random_seed
from cakechat.utils.files_utils import is_non_empty_file
from cakechat.utils.logger import WithLogger


class KerasTFModelIsolator(object):
    def __init__(self):
        # Use global keras (tensorflow) session config here
        keras_session_config = K.get_session()._config

        self._keras_isolated_graph = K.tf.Graph()
        self._keras_isolated_session = K.tf.Session(graph=self._keras_isolated_graph, config=keras_session_config)

    def _isolate_func(self, func):
        def wrapper(*args, **kwargs):
            with self._keras_isolated_graph.as_default():
                with self._keras_isolated_session.as_default():
                    return func(*args, **kwargs)

        return wrapper


class EvaluateAndSaveBestIntermediateModelCallback(AbstractKerasModelCallback, WithLogger):
    def __init__(self, model, eval_state_per_batches):
        """
        :param model: AbstractKerasModel object
        :param eval_state_per_batches: run model evaluation each `eval_state_per_batches` steps
        """
        super(EvaluateAndSaveBestIntermediateModelCallback, self).__init__(model)
        WithLogger.__init__(self)

        self._eval_state_per_batches = eval_state_per_batches
        self._training_start_time = None
        self._cur_epoch_start_time = None

    @property
    def callback_params(self):
        return {'eval_state_per_batches': self._eval_state_per_batches}

    @property
    def runs_only_on_main_worker(self):
        return True

    @staticmethod
    def _get_formatted_time(seconds):
        return str(timedelta(seconds=int(seconds)))

    def _log_metrics(self, dataset_name_to_metrics):
        for dataset_name, metrics in dataset_name_to_metrics.items():
            for metric_name, metric_value in metrics.items():
                self._logger.info('{} {} = {}'.format(dataset_name, metric_name, metric_value))
                self._model.metrics_plotter.plot(self._model.model_id, '{}/{}'.format(dataset_name, metric_name),
                                                 metric_value)

    def _eval_and_save_current_model(self, batch_num=None):
        total_elapsed_time = time.time() - self._training_start_time
        self._logger.info('Total elapsed time: {}'.format(self._get_formatted_time(total_elapsed_time)))

        if batch_num:
            elapsed_time_per_batch = (time.time() - self._cur_epoch_start_time) / batch_num
            self._logger.info('Cur batch num: {}; Train time per batch: {:.2f} seconds'.format(
                batch_num, elapsed_time_per_batch))

        dataset_name_to_metrics = self._model._evaluate()
        self._log_metrics(dataset_name_to_metrics)

        if not os.path.exists(self._model.model_path):
            os.makedirs(self._model.model_path)

        if self._model.metrics is None or self._model._is_better_model(dataset_name_to_metrics, self._model.metrics):
            self._logger.info('Obtained new best model. Saving it to {}'.format(self._model._model_resource_path))
            self._model._save_model(self._model._model_resource_path)
            self._model._metrics_serializer.save_metrics(self._model._metrics_resource_path, dataset_name_to_metrics)
            self._model._metrics = dataset_name_to_metrics

        self._model._save_model(self._model._model_progress_resource_path)

    def on_train_begin(self, logs=None):
        self._logger.info('Start training')
        self._training_start_time = time.time()

    def on_train_end(self, logs=None):
        self._logger.info('Stop training and compute final model metrics')
        self._eval_and_save_current_model()

    def on_batch_end(self, batch_num, logs=None):
        if batch_num > 0 and batch_num % self._eval_state_per_batches == 0:
            self._eval_and_save_current_model(batch_num)

    def on_epoch_begin(self, epoch_num, logs=None):
        cur_epoch_num = epoch_num + 1
        self._logger.info('Starting epoch {}'.format(cur_epoch_num))
        self._cur_epoch_start_time = time.time()

    def on_epoch_end(self, epoch_num, logs=None):
        cur_epoch_num = epoch_num + 1
        cur_epoch_time = time.time() - self._cur_epoch_start_time
        self._logger.info('For epoch {} elapsed time: {}'.format(cur_epoch_num,
                                                                 self._get_formatted_time(cur_epoch_time)))


class AbstractKerasModel(AbstractModel, metaclass=abc.ABCMeta):
    # Model resources default values
    _MODEL_PROGRESS_RESOURCE_NAME = 'model.current'

    def __init__(self, metrics_plotter=None, horovod=None, training_callbacks=None, *args, **kwargs):
        """
        :param metrics_plotter: object that plots training and validation metrics (see `TensorboardMetricsPlotter`)
        :param horovod: horovod module initialized for training on multiple GPUs. If None, uses single GPU, or CPU
        :param training_callbacks: list of instances of `AbstractKerasModelCallback`/`ParametrizedCallback` or None.
            In subclasses, please call `_create_essential_callbacks` to get essential callbacks, and/or put your own
            ones in this argument.
        """
        super(AbstractKerasModel, self).__init__(*args, **kwargs)

        self._metrics_plotter = metrics_plotter if metrics_plotter else DummyMetricsPlotter()
        self._horovod = horovod

        self._class_weight = None
        self._callbacks = training_callbacks or []

    @staticmethod
    def _create_essential_callbacks(model, horovod=None, eval_state_per_batches=None):
        """
        :param model: a model object, typically `self`
        :param horovod: if not None, adds callback for model params broadcasting between workers
        :param eval_state_per_batches: if not None, adds callback to evaluate the model every `eval_state_per_batches`
                batches
        :return: a list of callbacks
        """
        callbacks = []

        if horovod:
            callbacks.append(
                ParametrizedCallback(
                    horovod.callbacks.BroadcastGlobalVariablesCallback(0), runs_only_on_main_worker=False))

        if eval_state_per_batches:
            callbacks.append(EvaluateAndSaveBestIntermediateModelCallback(model, eval_state_per_batches))

        return callbacks

    def _get_worker_callbacks(self):
        if is_main_horovod_worker(self._horovod):
            # all callbacks should be run on main worker
            return self._callbacks

        # but not all callbacks should be run on a not main worker
        return [callback for callback in self._callbacks if not callback.runs_only_on_main_worker]

    @staticmethod
    def _to_keras_callbacks(callbacks):
        """
        Casts AbstractKerasModel callbacks (see `cakechat.dialog_model.callbacks`) to the keras-based ones (instances of
            `keras.callbacks.Callback`)
        :param callbacks:
        :return:
        """
        keras_callbacks = []
        for custom_callback in callbacks:
            if isinstance(custom_callback, AbstractKerasModelCallback):
                keras_callback = _KerasCallbackAdapter(custom_callback)
            elif isinstance(custom_callback, ParametrizedCallback):
                keras_callback = custom_callback.callback
            else:
                raise ValueError('Unsupported callback type: {}'.format(type(custom_callback)))

            keras_callbacks.append(keras_callback)

        return keras_callbacks

    def _set_class_weight(self, class_weight):
        self._class_weight = class_weight

    @property
    @abc.abstractmethod
    def _model_params(self):
        pass

    @property
    def model_params(self):
        params = {
            'training_callbacks': {
                cb.__class__.__name__: cb.callback_params for cb in self._callbacks if cb.callback_params
            }
        }
        params.update(self._model_params)
        return params

    @property
    def _model_progress_resource_path(self):
        return os.path.join(self.model_path, self._MODEL_PROGRESS_RESOURCE_NAME)

    @property
    def model(self):
        self.init_model()
        return self._model

    @property
    def metrics_plotter(self):
        return self._metrics_plotter

    @abc.abstractmethod
    def _get_training_model(self):
        pass

    @abc.abstractmethod
    def _build_model(self):
        pass

    @abc.abstractmethod
    def _is_better_model(self, new_metrics, old_metrics):
        pass

    @abc.abstractmethod
    def _get_training_batch_generator(self):
        """
        :return: generator with (inputs, targets) or (inputs, targets, sample_weights) tuples.
        The generator is expected to loop over its data indefinitely.
        An epoch finishes when epoch_batches_num batches have been seen by the training worker.
        """
        pass

    @abc.abstractmethod
    def _get_epoch_batches_num(self):
        pass

    def _save_model(self, model_file_path):
        self._model.save(model_file_path, overwrite=True)
        self._logger.info('Saved model weights to {}'.format(model_file_path))

    def _load_model(self, fresh_model, model_file_path):
        fresh_model.load_weights(model_file_path, by_name=True)
        self._logger.info('Restored model weights from {}'.format(model_file_path))
        return fresh_model

    def _load_model_if_exists(self):
        if is_non_empty_file(self._model_progress_resource_path):
            self._model = self._load_model(self._model, self._model_progress_resource_path)
            self._metrics = self._metrics_serializer.load_metrics(self._metrics_resource_path)
            return

        self._logger.info('Could not find saved model at {}\nModel will be trained from scratch.\n'
                          .format(self._model_progress_resource_path))

    def print_weights_summary(self):
        summary = '\n\nModel weights summary:'
        summary += '\n\t{0:<80} {1:<20} {2:}\n'.format('layer name', 'output shape:', 'size:')

        weights_names = [weight.name for layer in self._model.layers for weight in layer.weights]
        weights = self._model.get_weights()

        total_network_size = 0
        for name, weight in zip(weights_names, weights):
            param_size = weight.nbytes / 1024 / 1024
            summary += '\n\t{0:<80} {1:20} {2:<.2f}Mb'.format(name, str(weight.shape), param_size)
            total_network_size += param_size

        summary += '\n\nTotal network size: {0:.1f} Mb\n'.format(total_network_size)
        self._logger.info(summary)

    def init_model(self):
        if not self._model:
            self._logger.info('Initializing NN model')
            self._model = self._build_model()
            self._logger.info('NN model is initialized\n')
            self.print_weights_summary()

    def train_model(self):
        self.init_model()
        self._load_model_if_exists()

        set_horovod_worker_random_seed(self._horovod)
        training_batch_generator = self._get_training_batch_generator()

        epoch_batches_num = self._get_epoch_batches_num()
        workers_num = self._horovod.size() if self._horovod else 1

        self._logger.info('Total epochs num = {}; Total batches per epochs = {}; Total workers for train = {}'.format(
            self.model_params['epochs_num'], epoch_batches_num, workers_num))

        worker_callbacks = self._get_worker_callbacks()
        training_model = self._get_training_model()
        training_model.fit_generator(
            training_batch_generator,
            steps_per_epoch=epoch_batches_num // workers_num,
            callbacks=self._to_keras_callbacks(worker_callbacks),
            epochs=self.model_params['epochs_num'],
            class_weight=self._class_weight,
            verbose=0,
            workers=0)

        # reload model with the best quality
        if is_main_horovod_worker(self._horovod):
            self._model = self._load_model(self._model, self._model_resource_path)
