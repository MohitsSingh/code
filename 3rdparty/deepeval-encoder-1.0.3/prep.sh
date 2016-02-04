#!/bin/bash

# TODO: adjust the paths to CUDA, MKL, and libstdc++ as required

# CUDA (not required for the CPU backend, uncomment if using the CUDA backend)
#CUDA_HOME="/usr/local/cuda"

#export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${CUDA_HOME}/lib64:${CUDA_HOME}/lib"
#export PATH=${CUDA_HOME}/bin:${PATH}

# MKL
MKL_DIR="/opt/intel/mkl"
MKL_LIB_DIR="${MKL_DIR}/lib:${MKL_DIR}/lib/intel64"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:${MKL_LIB_DIR}"

# libstdc++
export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6
# export LD_PRELOAD=/usr/lib64/libstdc++.so.6
