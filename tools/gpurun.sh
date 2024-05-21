#!/bin/bash

gpu=$((${PMIX_RANK}))
echo ${PMIX_RANK} gpu $gpu
export CUDA_VISIBLE_DEVICES=$gpu
"$@"
