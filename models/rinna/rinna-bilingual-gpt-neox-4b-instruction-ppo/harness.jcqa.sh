MODEL_ARGS="pretrained=rinna/bilingual-gpt-neox-4b-instruction-ppo,use_fast=False,device_map=auto,torch_dtype=auto"
TASK="jcommonsenseqa-1.1-0.5"
python main.py --model hf-causal --model_args $MODEL_ARGS --tasks $TASK --num_fewshot "3" --device "cuda" --output_path "models/rinna/rinna-bilingual-gpt-neox-4b/result.jcqa.json"
