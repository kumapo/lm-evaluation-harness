#!/bin/bash
MODEL_ARGS="pretrained=stabilityai/japanese-stablelm-instruct-alpha-7b,use_fast=False,trust_remote_code=True,device_map=auto,torch_dtype=auto,load_in_8bit=True,offload_folder=/tmp,tokenizer=novelai/nerdstash-tokenizer-v1,additional_special_tokens=['▁▁']"
TASK="jcommonsenseqa-1.1-0.3"
NUM_FEWSHOT="3"
OUTPUT_PATH="models/stabilityai/japanese-stablelm-instruct-alpha-7b/result.jcqa.json"
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot $NUM_FEWSHOT \
    --device "cuda" \
    --no_cache \
    --output_path $OUTPUT_PATH
