#!/bin/bash
gpu=$((${OMPI_COMM_WORLD_LOCAL_RANK}%4))
echo `hostname` rank ${PMIX_RANK} gpu $gpu
export CUDA_VISIBLE_DEVICES=$gpu
"$@"
