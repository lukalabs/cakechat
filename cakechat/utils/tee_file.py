import os
import pickle
import tempfile

# Pickle on HIGHEST_PROTOCOL breaks on Python 3.6.5
_PICKLE_PROTOCOL = 2


def _pickle_iterable(filename, iterable):
    with open(filename, 'wb') as pickle_fh:
        pklr = pickle.Pickler(pickle_fh, _PICKLE_PROTOCOL)
        for entry in iterable:
            pklr.dump(entry)
            pklr.clear_memo()


def _open_pickle(filename):
    return open(filename, 'rb')


def _unpickle_iterable(pickle_fh):
    with pickle_fh:
        unpklr = pickle.Unpickler(pickle_fh)
        try:
            while True:
                yield unpklr.load()
        except EOFError:
            pass


def file_buffered_tee(iterable, n=2):
    _, filename = tempfile.mkstemp()
    try:
        _pickle_iterable(filename, iterable)
        return tuple(_unpickle_iterable(_open_pickle(filename)) for _ in range(n))
    finally:
        os.remove(filename)
