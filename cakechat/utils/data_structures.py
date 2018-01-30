import itertools
import collections


def flatten(xs, constructor=list):
    return constructor(itertools.chain.from_iterable(xs))


def create_namedtuple_instance(name, **kwargs):
    return collections.namedtuple(name, kwargs.keys())(**kwargs)
