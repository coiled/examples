ARG CUDA_VER=12.9.0
ARG PYTHON_VER=3.12
ARG LINUX_VER=ubuntu20.04

FROM rapidsai/miniforge-cuda:cuda${CUDA_VER}-base-${LINUX_VER}-py${PYTHON_VER} as base
ARG CUDA_VER
ARG PYTHON_VER

WORKDIR /home
COPY pytorch.yml pytorch.yml
RUN mamba env update -n base --file pytorch.yml
RUN conda clean -afy
RUN conda uninstall -y pytorch torchvision
RUN mamba install -y -n base -c conda-forge pytorch-gpu torchvision
RUN conda clean -afy
RUN rm pytorch.yml


FROM base as examples
COPY . .
