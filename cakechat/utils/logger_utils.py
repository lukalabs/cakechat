import logging


class DefaultFormatter(logging.Formatter):
    _FMT = '[%(asctime)s.%(msecs)03d][%(levelname)s][%(process)s][%(name)s][%(lineno)d] %(message)s'
    _DATEFMT = '%d.%m.%Y %H:%M:%S'

    def __init__(self):
        super(DefaultFormatter, self).__init__(fmt=self._FMT, datefmt=self._DATEFMT)


class FormattedStreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super(FormattedStreamHandler, self).__init__(stream)
        self.formatter = DefaultFormatter()


class LaconicFormatter(logging.Formatter):
    _FMT = '%(message)s'

    def __init__(self):
        super(LaconicFormatter, self).__init__(fmt=self._FMT)


class LaconicStreamHandler(logging.StreamHandler):
    def __init__(self, stream=None):
        super(LaconicStreamHandler, self).__init__(stream)
        self.formatter = LaconicFormatter()
