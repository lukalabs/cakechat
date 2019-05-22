import json

from cakechat.utils.files_utils import ensure_file
from cakechat.utils.logger import get_logger

_logger = get_logger(__name__)


class MetricsException(Exception):
    pass


class MetricsSerializer(object):
    @staticmethod
    def load_metrics(metrics_resource_name):
        _logger.info('Restoring metrics from {}'.format(metrics_resource_name))
        with open(metrics_resource_name, 'r', encoding='utf-8') as fh:
            return json.load(fh)

    @staticmethod
    def save_metrics(metrics_resource_name, metrics):
        _logger.info('Saving metrics to {}'.format(metrics_resource_name))
        with ensure_file(metrics_resource_name, 'w', encoding='utf-8') as fh:
            json.dump(metrics, fh, indent=2)
