#!/bin/bash
set -eu
MODEL_ARGS="pretrained=stabilityai/japanese-stablelm-base-alpha-7b,use_fast=False,trust_remote_code=True,device_map=auto,torch_dtype=auto,offload_folder=/tmp"
TASK="jcommonsenseqa-1.2-0.2"
NUM_FEW_SHOTS="3"
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot $NUM_FEW_SHOTS \
    --device "cuda" \
    --output_path "models/stablelm/stablelm-ja-base-alpha-7b/result.jcqa-1.2.json"
