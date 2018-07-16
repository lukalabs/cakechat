from collections import namedtuple


Dataset = namedtuple('Dataset', ['x', 'y', 'condition_ids'])

DatasetsCollection = namedtuple('DatasetsCollection', [
    'train',
    'train_subset',
    'context_free_val',
    'context_sensitive_val',
    'context_sensitive_val_subset'
])

TrainStats = namedtuple('TrainStats', [
    'cur_batch_id',
    'batches_num',
    'start_time',
    'total_training_time',
    'cur_loss',
    'best_val_perplexities',
    'cur_val_metrics'
])
