# FROM ubuntu:18.04 comment out
FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y tzdata
# timezone setting
ENV TZ=Asia/Tokyo 
# pythonとpipちゃん
RUN apt update && apt install -y python3-pip
RUN apt install python3
RUN apt-get remove python-pip python3-pip -y && apt-get install -y wget
# opencv-devのインストール
RUN apt-get update -y && apt-get install -y libopencv-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN apt update -y  && apt install curl -y
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py
RUN python3 get-pip.py
# TensorflowとOpencvのインストール
RUN pip3 install numpy tensorflow-gpu opencv-python

# kerasとsklearnのインストール
RUN pip3 install keras
RUN pip3 install scikit-learn
RUN pip3 install pillow

ENV APP_NAME tensor-docker
WORKDIR /home/$APP_NAME

# Install prerequisite packages
RUN apt-get update \
 && apt-get install -yq --no-install-recommends \
      jupyter-notebook

# User configuration
ARG USERNAME=jupyter
RUN useradd -m -s /bin/bash ${USERNAME}
USER ${USERNAME}

# Jupyter configuration
RUN jupyter notebook --generate-config \
 && mkdir -p /home/${USERNAME}/jupyter-working \
 && sed -i.back \
    -e "s:^#c.NotebookApp.token = .*$:c.NotebookApp.token = u'':" \
    -e "s:^#c.NotebookApp.ip = .*$:c.NotebookApp.ip = '*':" \
    -e "s:^#c.NotebookApp.open_browser = .*$:c.NotebookApp.open_browser = False:" \
    -e "s:^#c.NotebookApp.notebook_dir = .*$:c.NotebookApp.notebook_dir = '/home/${USERNAME}/jupyter-working':" \
    /home/${USERNAME}/.jupyter/jupyter_notebook_config.py

RUN pip3 install matplotlib

# Expose container ports
EXPOSE 8888

# Boot process
CMD jupyter notebook
