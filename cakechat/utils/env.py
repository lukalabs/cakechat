import os


def _use_gpu_env():
    try:
        use_gpu = os.environ['USE_GPU']
        return int(use_gpu)
    except (KeyError, ValueError):
        return None


def is_dev_env():
    try:
        is_dev = os.environ['IS_DEV']
        return bool(int(is_dev))
    except (KeyError, ValueError):
        return False


def _init_cuda_env():
    # Set GPU device order the same as in nvidia-smi
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'


def init_theano_env(gpu_id=_use_gpu_env(), cnmem=0, float_precision='float32', is_dev=is_dev_env()):
    """
    :param gpu_id: ID of GPU to use, default is None (No GPU support, CPU-only);
    :param cnmem: The value represents the start size (either in MB or the fraction of total GPU memory) of the memory
        pool. Default: 0 (Preallocation of size 0, only cache the allocation)
    :param float_precision: String specifying floating point precision. Can be 'float64', 'float32', or 'float16'
    :param is_dev: Apply just a few graph optimizations and only use Python implementations. Default is False.
        GPU is disabled, CPU only. Drastically speeds up theano graph compilation. Use for development purposes.
    :return:
    """
    _init_cuda_env()

    theano_flags = 'floatX={}'.format(float_precision)

    if is_dev:
        # Use fast_compile only in dev-env because it doesn't works on GPU with libgpuarray
        theano_flags += ',device=cpu,mode=FAST_COMPILE'
    elif gpu_id is None:
        theano_flags += ',device=cpu'
    else:
        theano_flags += ',device=cuda{},gpuarray.preallocate={:0.2}'.format(gpu_id, float(cnmem))

    if 'THEANO_FLAGS' in os.environ:
        os.environ['THEANO_FLAGS'] = theano_flags + ',' + os.environ['THEANO_FLAGS']
    else:
        os.environ['THEANO_FLAGS'] = theano_flags
