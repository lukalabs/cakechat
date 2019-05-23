import json
from copy import copy

from cakechat.utils.logger import get_logger

_logger = get_logger(__name__)


class FileTextLinesIterator(object):
    def __init__(self, filename):
        self._filename = filename

    def __iter__(self):
        for line in open(self._filename, 'r', encoding='utf-8'):
            yield line.strip()

    def __copy__(self):
        return FileTextLinesIterator(self._filename)


class ProcessedLinesIterator(object):
    def __init__(self, lines_iter, processing_callbacks=None):
        self._lines_iter = lines_iter
        self._processing_callbacks = processing_callbacks if processing_callbacks else []

    def __iter__(self):
        for line in self._lines_iter:
            for callback in self._processing_callbacks:
                line = callback(line)
            yield line

    def __copy__(self):
        return ProcessedLinesIterator(copy(self._lines_iter), self._processing_callbacks)


class JsonTextLinesIterator(object):
    def __init__(self, text_lines_iter):
        self._text_lines_iter = text_lines_iter

    def __iter__(self):
        for line in self._text_lines_iter:
            try:
                yield json.loads(line.strip())
            except ValueError:
                _logger.warning('Skipped invalid json object: "{}"'.format(line.strip()))
                continue

    def __copy__(self):
        return JsonTextLinesIterator(copy(self._text_lines_iter))
