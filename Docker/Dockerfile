FROM pytorch/pytorch:1.6.0-cuda10.1-cudnn7-devel
WORKDIR /studio

# download packages
RUN mv /etc/apt/sources.list.d /etc/apt/temp.sources.d
RUN apt-get update

RUN apt-get install -y \
    openssh-server \
    tmux \
    vim \
    proxychains \
    git 

# config conda
RUN conda clean -i
RUN conda init bash

# create links
RUN ln -s /studio ~/studio

