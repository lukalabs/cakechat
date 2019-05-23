# Dockerfile for CPU-only CakeChat setup

FROM ubuntu:18.04

ENV LANG C.UTF-8

# Install some dependencies
RUN apt-get update && apt-get install -y \
        curl \
        git \
        screen \
        tmux \
        sudo \
        nano \
        pkg-config \
        software-properties-common \
        unzip \
        vim \
        wget \
        python3 \
        python3-dev \
        python3-pip \
        python3-setuptools \
        && \
    apt-get clean && \
    apt-get autoremove && \
    rm -rf /var/lib/apt/lists/*

# Link python to python3 (since python 2 is used by default in ubuntu docker image)
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install up-to-date pip
RUN pip3 --no-cache-dir install -U pip

# setup cakechat and install dependencies
RUN git clone https://github.com/lukalabs/cakechat.git /root/cakechat
RUN pip3 --no-cache-dir install -r /root/cakechat/requirements.txt -r /root/cakechat/requirements-local.txt
RUN mkdir -p /root/cakechat/data/tensorboard

WORKDIR /root/cakechat
CMD git pull && \
    pip3 install -r requirements.txt -r /root/cakechat/requirements-local.txt && \
    (tensorboard --logdir=data/tensorboard 2>data/tensorboard/err.log &); \
    /bin/bash
