MODEL_ARGS="pretrained=rinna/japanese-gpt-neox-3.6b-instruction-ppo,use_fast=False,device_map=auto,torch_dtype=auto"
TASK="jcommonsenseqa-1.1-0.4"
python main.py --model hf-causal --model_args $MODEL_ARGS --tasks $TASK --num_fewshot "3" --device "cuda" --output_path "models/rinna/rinna-japanese-gpt-neox-3.6b-instruction-ppo/result.jcqa.json"
