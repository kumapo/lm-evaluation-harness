MODEL_ARGS="pretrained=cyberagent/open-calm-3b,device_map=auto,torch_dtype=auto,load_in_8bit=True,low_cpu_mem_usage=True"
TASK="jcommonsenseqa-1.2-0.2"
python main.py --model hf-causal --model_args $MODEL_ARGS --tasks $TASK --num_fewshot "3" --device "cuda" --output_path "models/cyberagent/cyberagent-open-calm-3b/result.jcqa-1.2.json"
