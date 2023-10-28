MODEL_ARGS="pretrained=meta-llama/Llama-2-7b-chat-hf,use_accelerate=True,dtype=auto"
TASK="jcommonsenseqa-1.1-0.3"
python main.py --model hf-causal-experimental --model_args $MODEL_ARGS --tasks $TASK --num_fewshot "3" --device "cuda" --output_path "models/llama2/llama2-7b-chat/result.jcqa.json" --batch_size 2
