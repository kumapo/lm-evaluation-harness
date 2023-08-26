MODEL_ARGS="pretrained=rinna/japanese-gpt-neox-3.6b-instruction-ppo,use_fast=False,device_map=auto,torch_dtype=auto"
TASK="jsquad-1.1-0.41"
python main.py --model hf-causal --model_args $MODEL_ARGS --tasks $TASK --num_fewshot "2" --device "cuda" --output_path "models/rinna/rinna-japanese-gpt-neox-3.6b-instruction-ppo/result.jsquad-1.1-0.41.json"
