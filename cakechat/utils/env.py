import os
import subprocess

import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session

from cakechat.utils.logger import get_logger

_logger = get_logger(__name__)


def is_dev_env():
    try:
        is_dev = os.environ['IS_DEV']
        return bool(int(is_dev))
    except (KeyError, ValueError):
        return False


def init_cuda_env():
    os.environ['PATH'] += ':/usr/local/cuda/bin'
    os.environ['LD_LIBRARY_PATH'] = '/usr/local/cuda/lib64:/usr/local/nvidia/lib64/:/usr/local/cuda/extras/CUPTI/lib64'
    os.environ['LIBRARY_PATH'] = '/usr/local/share/cudnn'
    os.environ['CUDA_HOME'] = '/usr/local/cuda'
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


def try_import_horovod():
    try:
        import horovod.keras as hvd
    except ImportError:
        return None
    else:
        return hvd


def init_keras(hvd=None):
    """
    Set config for Horovod. Config params copied from official example:
    https://github.com/uber/horovod/blob/master/examples/keras_mnist_advanced.py#L15

    :param hvd: instance of horovod.keras
    """

    init_cuda_env()
    config = tf.ConfigProto()

    if hvd:
        hvd.init()
        config.gpu_options.allow_growth = True
        config.gpu_options.visible_device_list = str(hvd.local_rank())

    set_session(tf.Session(config=config))


def set_keras_tf_session(gpu_memory_fraction):
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = float(gpu_memory_fraction)  # pylint: disable=maybe-no-member
    set_session(tf.Session(config=config))


def run_horovod_train(train_cmd, gpu_ids):
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(gpu_ids)

    cmd = 'mpirun -np {workers_nums} -H localhost:{workers_nums} {train_cmd}'.format(
        workers_nums=len(gpu_ids), train_cmd=train_cmd)
    process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    while process.poll() is None:
        output = process.stdout.readline()
        if output:
            print(output.strip())


def is_main_horovod_worker(horovod):
    return horovod is None or horovod.rank() == 0


def set_horovod_worker_random_seed(horovod):
    seed = horovod.rank() if horovod else 0
    np.random.seed(seed)

