import os
from collections import Counter
import tensorflow as tf
from keras import backend as K

from cakechat.utils.files_utils import get_cached, serialize


class DummyMetricsPlotter(object):
    def plot(self, model_id, metric_name, metric_value):
        pass


class TensorboardMetricsPlotter(object):
    def __init__(self, log_dir):
        self._log_dir = log_dir
        self._writers = {}
        self._steps_path = os.path.join(self._log_dir, 'steps')
        self._steps = get_cached(Counter, self._steps_path)

    @staticmethod
    def _get_model_specific_key(model_name, key):
        """
        Build unique identifier for (model_name, key_name) pair.
        """
        return '{}_{}'.format(model_name, key)

    def _get_model_writer(self, model_name):
        if model_name not in self._writers:
            self._writers[model_name] = \
                tf.summary.FileWriter(os.path.join(self._log_dir, model_name), K.get_session().graph)

        return self._writers[model_name]

    def plot(self, model_name, metric_name, metric_value):
        summary = tf.Summary()
        summary.value.add(tag=metric_name, simple_value=metric_value)  # pylint: disable=maybe-no-member

        writer = self._get_model_writer(model_name)
        metric_model_key = self._get_model_specific_key(model_name, metric_name)
        writer.add_summary(summary, self._steps[metric_model_key])
        writer.flush()

        self._steps[metric_model_key] += 1
        serialize(self._steps_path, self._steps)

    def log_run_metadata(self, model_name, run_metadata):
        run_metadata_model_key = self._get_model_specific_key(model_name, key='run_metadata')
        run_tag = '{}_{}'.format(self._steps[run_metadata_model_key], run_metadata_model_key)

        writer = self._get_model_writer(model_name)
        writer.add_run_metadata(run_metadata, run_tag, self._steps[run_metadata_model_key])
        writer.flush()

        self._steps[run_metadata_model_key] += 1
        serialize(self._steps_path, self._steps)

    @property
    def log_dir(self):
        return self._log_dir
