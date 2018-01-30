import os
from collections import Counter

from tensorboard_logger import Logger as TensorboardLogger

from cakechat.utils.files_utils import get_persisted, serialize


class TensorboardMetricsPlotter(object):
    def __init__(self, log_dir):
        self._log_dir = log_dir
        self._writers = {}
        self._steps_path = os.path.join(log_dir, 'steps')
        self._steps = get_persisted(dict, self._steps_path)

    def plot(self, model_name, metric_name, metric_value):
        if model_name not in self._writers:
            self._writers[model_name] = TensorboardLogger(os.path.join(self._log_dir, model_name))
        if model_name not in self._steps:
            self._steps[model_name] = Counter()

        self._writers[model_name].log_value(metric_name, metric_value, step=self._steps[model_name][metric_name])
        self._steps[model_name][metric_name] += 1

        serialize(self._steps_path, self._steps)
