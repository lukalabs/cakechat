import time
from functools import wraps

from cakechat.utils.logger import get_logger

_logger = get_logger(__name__)


def _execute_and_profile(fn, *args, **kwargs):
    start_time = time.time()
    fn_result = fn(*args, **kwargs)
    execution_time = time.time() - start_time

    _logger.info('Elapsed time for "{}": {}'.format(fn.__name__, execution_time))
    return execution_time, fn_result


def timer(fn):
    """
    Timer decorator. Logs execution time of the function.
    """

    @wraps(fn)
    def _perform(*args, **kwargs):
        _, fn_result = _execute_and_profile(fn, *args, **kwargs)
        return fn_result

    return _perform
