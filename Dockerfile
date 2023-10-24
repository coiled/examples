ARG CUDA_VER=11.8.0
ARG PYTHON_VER=3.10
ARG LINUX_VER=ubuntu20.04

FROM rapidsai/miniforge-cuda:cuda${CUDA_VER}-base-${LINUX_VER}-py${PYTHON_VER} as base
ARG CUDA_VER
ARG PYTHON_VER

WORKDIR /home
COPY pytorch.yml pytorch.yml
RUN mamba env update -n base --file pytorch.yml \
    && conda clean -afy \
    && mamba uninstall -y pytorch torchvision \
    && mamba install -y -n base -c pytorch -c nvidia -c conda-forge \
        "cudatoolkit=${CUDA_VER%.*}.*" \
        "cuda-version=${CUDA_VER%.*}.*" \
        pytorch \
        torchvision \
        "pytorch-cuda=${CUDA_VER%.*}.*" \
    && conda clean -afy \
    && rm pytorch.yml


FROM base as examples
COPY . .
