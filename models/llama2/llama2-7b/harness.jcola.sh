MODEL_ARGS="pretrained=meta-llama/Llama-2-7b-hf,use_accelerate=True,dtype=auto"
TASK="jcola-0.1-0.3"
python main.py --model hf-causal-experimental --model_args $MODEL_ARGS --tasks $TASK --num_fewshot "3" --device "cuda" --output_path "models/llama2/llama2-7b/result.jcola.json" --batch_size 2
