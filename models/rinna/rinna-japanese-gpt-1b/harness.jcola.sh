MODEL_ARGS="pretrained=rinna/japanese-gpt-1b,use_fast=False"
TASK="jcola-0.1-0.1"
python main.py --model hf-causal --model_args $MODEL_ARGS --tasks $TASK --num_fewshot "2" --device "cuda" --output_path "models/rinna/rinna-japanese-gpt-1b/result.jcola.json"
