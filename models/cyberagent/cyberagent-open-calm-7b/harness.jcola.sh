MODEL_ARGS="pretrained=cyberagent/open-calm-7b,device_map=auto,torch_dtype=auto"
TASK="jcola-0.1-0.1"
python main.py --model hf-causal --model_args $MODEL_ARGS --tasks $TASK --num_fewshot "3" --device "cuda" --output_path "models/cyberagent/cyberagent-open-calm-7b/result.jcola.json"
