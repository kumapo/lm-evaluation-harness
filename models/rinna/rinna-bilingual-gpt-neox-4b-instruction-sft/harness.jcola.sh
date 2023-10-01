MODEL_ARGS="pretrained=rinna/bilingual-gpt-neox-4b-instruction-sft,use_fast=False,device_map=auto,torch_dtype=auto"
TASK="jcola-0.1-0.5"
python main.py --model hf-causal --model_args $MODEL_ARGS --tasks $TASK --num_fewshot "3" --device "cuda" --output_path "models/rinna/rinna-bilingual-gpt-neox-4b-instruction-sft/result.jcola.json"
