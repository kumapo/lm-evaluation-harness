MODEL_ARGS="pretrained=cyberagent/open-calm-3b"
TASK="jcommonsenseqa-1.1-0.2.1"
python main.py --model hf-causal --model_args $MODEL_ARGS --tasks $TASK --num_fewshot "3" --device "cuda" --output_path "models/cyberagent/cyberagent-open-calm-3b/result.jcqa.json"
