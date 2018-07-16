import logging
import logging.config

logging.config.dictConfig({
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'cakechat.utils.logger_utils.FormattedStreamHandler',
            'level': 'INFO'
        },
        'laconic': {
            'class': 'cakechat.utils.logger_utils.LaconicStreamHandler',
            'level': 'INFO'
        }
    },
    'loggers': {
        'cakechat': {
            'handlers': ['console'],
            'level': 'INFO',
        },
        'cakechat.laconic_logger': {
            'handlers': ['laconic'],
            'level': 'INFO',
            'propagate': False
        }
    }
})


def get_logger(name):
    return logging.getLogger(name)


def get_tools_logger(name):
    return logging.getLogger('cakechat.' + name)


def _get_laconic_logger():
    return get_tools_logger('laconic_logger')


class WithLogger(object):
    def __init__(self):
        self._logger = logging.getLogger(self.__class__.__module__ + '.' + self.__class__.__name__)


laconic_logger = _get_laconic_logger()
