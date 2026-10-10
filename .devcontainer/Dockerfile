# GPU development image for tenferro-decision-rs.
# tenferro-gpu's `cuda` feature loads NVRTC / cuBLAS / the runtime with
# cudarc's dynamic loading, and dlopens cuTENSOR (`libcutensor.so.2`), so these
# must be present at run time. libcuda comes from the host driver through the
# NVIDIA container runtime (a CUDA 13 driver, >= 580, is required).
FROM nvidia/cuda:13.0.3-devel-ubuntu24.04

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential ca-certificates clang curl git git-lfs less pkg-config \
        libcutensor2-cuda-13 python3 sudo \
    && rm -rf /var/lib/apt/lists/* \
    && echo "ubuntu ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/ubuntu

# The base image's `ubuntu` user has UID/GID 1000, matching the usual host user.
USER ubuntu
ENV PATH=/home/ubuntu/.cargo/bin:$PATH
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
        | sh -s -- -y --profile minimal --default-toolchain stable \
            --component rustfmt,clippy,rust-src \
    && mkdir -p /home/ubuntu/.cache/huggingface
