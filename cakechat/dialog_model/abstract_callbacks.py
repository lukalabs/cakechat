"""
Essentials for using training callbacks together with AbstractKerasModel.

TL;DR
1. If you are implementing your own callback, please inherit it from `AbstractKerasModelCallback`.
2. If you are using a stock (keras) callback, please wrap it with `ParametrizedCallback`
"""
import abc

import keras


class _KerasCallbackAdapter(keras.callbacks.Callback):
    """
    Class that adapts `AbstractKerasModelCallback`-based callback to the native keras one. Not assumed to be used
    directly in the client code.
    """

    def __init__(self, callback):
        """
        :param callback: instance of `AbstractKerasModelCallback`
        """
        super(_KerasCallbackAdapter, self).__init__()
        self.__callback = callback

    def on_epoch_begin(self, epoch, logs=None):
        return self.__callback.on_epoch_begin(epoch, logs)

    def on_epoch_end(self, epoch, logs=None):
        return self.__callback.on_epoch_end(epoch, logs)

    def on_batch_begin(self, batch, logs=None):
        return self.__callback.on_batch_begin(batch, logs)

    def on_batch_end(self, batch, logs=None):
        return self.__callback.on_batch_end(batch, logs)

    def on_train_begin(self, logs=None):
        return self.__callback.on_train_begin(logs)

    def on_train_end(self, logs=None):
        return self.__callback.on_train_end(logs)


class _AbstractCallback(object, metaclass=abc.ABCMeta):
    """
    Common interface for training callbacks used with AbstractKerasModel models
    """

    @property
    @abc.abstractmethod
    def callback_params(self):
        """
        :return dict of params that affect the resulting model
        """
        pass

    @property
    @abc.abstractmethod
    def runs_only_on_main_worker(self):
        """
        :return True, if this callback runs only on main worker (in case of distributed training), False otherwise
        """
        pass


class AbstractKerasModelCallback(_AbstractCallback, metaclass=abc.ABCMeta):
    """
    Base callback class that is compatible with `AbstractKerasModel` (so it can be used within keras via
    `_KerasCallbackAdapter`). If you are implementing your own callback that utilizes `AbstractKerasModel`, please
    inherit from this class and not from `keras.callbacks.Callback` in order to isolate your callback's variables and
    methods from the ones that belong to `keras.callbacks.Callback`.
    See `EvaluateAndSaveBestIntermediateModelCallback` as an example.
    """

    def __init__(self, model):
        """
        :param model: instance of `AbstractKerasModel`
        """
        super(AbstractKerasModelCallback, self).__init__()
        self._model = model

    def on_epoch_begin(self, epoch, logs=None):
        pass

    def on_epoch_end(self, epoch, logs=None):
        pass

    def on_batch_begin(self, batch, logs=None):
        pass

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_begin(self, logs=None):
        pass

    def on_train_end(self, logs=None):
        pass


class ParametrizedCallback(_AbstractCallback):
    """
    Provides `_AbstractCallback` interface for arbitrary callback object.
    If you are going to use one of the stock keras callbacks, please choose one of the following options:
        1. Use this class to instantiate a AbstractKerasModel-friendly callback object. Specify `callback_params` if the callback
            affects the resulting model by some parameters
        2. Create class <YourCallbackName>(ParametrizedCallback) and pass the original callback object, as well as
            `callback_params` if the callback affects the resulting model. Recommended, if you are going to publish your
            callback in the repository.
    See `AbstractKerasModel#_create_essential_callbacks` as an example.
    """

    def __init__(self, callback, runs_only_on_main_worker, callback_params=None):
        """
        :param callback: arbitrary callback object (e.g. instance of `keras.callbacks.Callback`)
        :param runs_only_on_main_worker: True, if this callback runs only on main worker (in case of distributed
                training), False otherwise
        :param callback_params: dict of params that affect the resulting model
        :return:
        """
        super(ParametrizedCallback, self).__init__()
        self._callback = callback
        self._callback_params = callback_params or {}
        self._runs_only_on_main_worker = runs_only_on_main_worker

    @property
    def callback_params(self):
        return self._callback_params

    @property
    def runs_only_on_main_worker(self):
        return self._runs_only_on_main_worker

    @property
    def callback(self):
        return self._callback
