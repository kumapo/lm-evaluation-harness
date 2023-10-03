MODEL_ARGS="pretrained=cyberagent/open-calm-medium,use_fast=True"
TASK="jcola-0.1-0.4"
python main.py --model hf-causal --limit 2 --model_args $MODEL_ARGS --tasks $TASK --num_fewshot "2" --device "cpu" --output_path "models/cyberagent-open-calm-medium/result.jsquad-1.2.json"
