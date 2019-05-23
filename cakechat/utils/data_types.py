from collections import namedtuple

Dataset = namedtuple('Dataset', ['x', 'y', 'condition_ids'])
ModelParam = namedtuple('ModelParam', ['value', 'id'])
