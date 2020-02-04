FROM nvidia/cuda:10.1-cudnn7-devel-ubuntu18.04
MAINTAINER caffe-maint@googlegroups.com

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ America/New_York
ENV CUDA_PATH /usr/local/cuda
ENV CAFFE_ROOT /opt/caffe/
RUN nvcc --version

#---- install libraries for Caffe
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        git \
        wget \
        libatlas-base-dev \
        libboost-all-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libopencv-dev \
        libprotobuf-dev \
        libsnappy-dev \
        protobuf-compiler \
        python-dev \
	python-setuptools \
        python-pip && \
	rm -rf /var/lib/apt/lists/*

# add necessary files to image
COPY lib /opt/lib/
COPY tools /opt/tools/
COPY caffe /opt/caffe/
COPY experiments /opt/experiments/
COPY models /opt/models/
COPY data /opt/data/
COPY features /opt/features/
COPY experiments /opt/experiments/

#--- install Python packages
WORKDIR $CAFFE_ROOT
#RUN pip install cmake==3.15.3

RUN for req in $(cat python/requirements.txt) pydot; do pip install $req; done && \
    mkdir build && cd build && \
    cmake -DUSE_CUDNN=1 .. && \
    make -j"$(nproc)" && \
    make install
RUN cd ../lib && make -j10
RUN cmake --version

# cmake -DUSE_CUDNN=1 -DCUDNN_INCLUDE="/usr/local/cuda-10.1/include" -DCUDNN_LIBRARY="/usr/local/cuda-10.1/lib64/libcudnn.so.7" .. && \
ENV PYCAFFE_ROOT $CAFFE_ROOT/python
ENV PYTHONPATH $PYCAFFE_ROOT:$PYTHONPATH
ENV PATH $CAFFE_ROOT/build/tools:$PYCAFFE_ROOT:$PATH
RUN echo "$CAFFE_ROOT/build/lib" >> /etc/ld.so.conf.d/caffe.conf && ldconfig

WORKDIR /workspace