FROM ubuntu:20.04
LABEL org.opencontainers.image.ref.name=ubuntu
LABEL org.opencontainers.image.version=20.04

ARG TARGETARCH=amd64

CMD ["/bin/bash"]

# install package
RUN apt-get update && \
    apt-get install -y --no-install-recommends curl vim sudo wget
RUN useradd -m winter && echo "winter:4958" | chpasswd && adduser winter sudo

# set CUDA
ENV NVARCH=x86_64
ENV NVIDIA_REQUIRE_CUDA=cuda>=11.3 brand=nvidia,driver>=470,driver<471 
ENV NV_CUDA_COMPAT_PACKAGE=cuda-compat-11-3
ENV NV_CUDA_CUDART_PACKAGE=cuda-cudart-11-3
RUN apt-get update && \
    apt-get install -y --no-install-recommends gnupg2 ca-certificates && \
    curl -fsSL https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/${NVARCH}/3bf863cc.pub | apt-key add - && \
    echo "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/${NVARCH} /" > /etc/apt/sources.list.d/cuda.list && \
    rm -rf /var/lib/apt/lists/* 
RUN apt-get update && \
    apt-get install -y --no-install-recommends ${NV_CUDA_CUDART_PACKAGE}\
    ${NV_CUDA_COMPAT_PACKAGE} && \
    rm -rf /var/lib/apt/lists/* 
RUN echo "/usr/local/nvidia/lib" >> /etc/ld.so.conf.d/nvidia.conf && \
    echo "/usr/local/nvidia/lib64" >> /etc/ld.so.conf.d/nvidia.conf 
ENV PATH=/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin
ENV LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
COPY ./copy/NGC-DL-CONTAINER-LICENSE /

# install python3, package-python, java
ENV PYTHONUNBUFFERED=TRUE
ARG PYTHON_VERSION=3.8
COPY ./copy/home/venv/bin /home/venv/bin 
RUN apt-get update --fix-missing && apt-get upgrade -y && \
    apt-get install -y software-properties-common && \
    add-apt-repository -y ppa:deadsnakes/ppa && \
    apt remove python-pip && \
	apt-get install -y python3-pip && \
    DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
    python$PYTHON_VERSION \
    python3-distutils \
    python$PYTHON_VERSION-dev \
    python$PYTHON_VERSION-venv \
    openjdk-17-jdk \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && cd /tmp 
# create symbolic link for use `python` command for `python3`
RUN ln -s /usr/bin/python3 /usr/local/bin/python

ENV PATH=/home/venv/bin:/usr/local/nvidia/bin:/usr/local/cuda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# COPY
RUN useradd -m model-server && \
    mkdir -p /home/model-server/tmp && \
    chown -R model-server /home/model-server && \
    mkdir /home/model-server/model-store && \
    chown -R model-server /home/model-server/model-store 
COPY ./copy/kserve_wrapper /home/model-server/kserve_wrapper
COPY ./copy/requirements.txt /home/model-server/requirements.txt
WORKDIR /home/model-server

# install requirements
RUN pip install -r requirements.txt
RUN pip install opencv-python-headless
RUN pip install torch==1.12.0+cu113  -f https://download.pytorch.org/whl/torch_stable.html

# set entryporint for kserve
COPY ./copy/dockerd-entrypoint.sh /usr/local/bin/dockerd-entrypoint.sh 
RUN chmod +x /usr/local/bin/dockerd-entrypoint.sh && \
    chmod +x /home/venv/bin

USER model-server
EXPOSE 7070/tcp 7071/tcp 8080/tcp 8081/tcp 8082/tcp
ENV TEMP=/home/model-server/tmp
ENTRYPOINT ["/usr/local/bin/dockerd-entrypoint.sh"]
CMD ["serve"]

