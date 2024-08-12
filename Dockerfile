# base image for development:
# Ubuntu
# CUDA + cuDNN

FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu20.04
USER root:root

ENV DEBIAN_FRONTEND=noninteractive

# # Configure User info.
# ARG USERNAME=dev
# ARG USER_UID=1002
# ARG USER_GID=$USER_UID
# RUN groupadd --gid $USER_GID $USERNAME \
#     && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME \
#     && apt-get update \
#     && apt-get install -y sudo \
#     && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
#     && chmod 0440 /etc/sudoers.d/$USERNAME

# General Settings
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64
ENV NCCL_DEBUG=INFO

# Install Common Dependencies
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    git \
    wget \
    vim \
    cmake && \
    # curl && \
    apt-get clean -y && \
    rm -rf /var/lib/apt/lists/*

# Conda Environment for Python
ENV PATH=/opt/miniconda/bin:$PATH
RUN wget -qO /tmp/miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash /tmp/miniconda.sh -bf -p /opt/miniconda && \
    conda clean -ay && \
    conda install pip && \
    rm -rf /opt/miniconda/pkgs && \
    chmod -R 777 /opt/miniconda && \
    rm /tmp/miniconda.sh && \
    find / -type d -name __pycache__ | xargs rm -rf

# Upgrade pip to latest
ENV PIP="pip install --no-cache-dir"
RUN ${PIP} --upgrade pip

USER dev
WORKDIR /workspace