# Development GPU-powered dockerfile

FROM horovod/horovod:0.16.0-tf1.12.0-torch1.0.0-mxnet1.4.0-py3.5

ENV LANG C.UTF-8

RUN apt-get update && apt-get install -y \
    cabextract \
    libcurl4-openssl-dev \
    libmemcached-dev \
    libmysqlclient-dev \
    zlib1g-dev \
    libssl-dev \
    sudo \
    htop \
    tmux \
    man \
    less

# Install up-to-date pip
RUN pip3 --no-cache-dir install -U pip

# setup cakechat and install dependencies
RUN git clone https://github.com/lukalabs/cakechat.git /root/cakechat
RUN pip3 --no-cache-dir install -r /root/cakechat/requirements.txt
RUN mkdir -p /root/cakechat/data/tensorboard

WORKDIR /root/cakechat
CMD git pull && \
    pip3 install -r requirements.txt && \
    (tensorboard --logdir=data/tensorboard 2>data/tensorboard/err.log &); \
    /bin/bash
