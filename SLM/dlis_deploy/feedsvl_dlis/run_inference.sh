CUDA_VISIBLE_DEVICES=0 nohup python3 inference_lora_with_reranker.py --cur_gpu_id 0 &
sleep 1s
CUDA_VISIBLE_DEVICES=1 nohup python3 inference_lora_with_reranker.py --cur_gpu_id 1 &
sleep 1s
CUDA_VISIBLE_DEVICES=2 nohup python3 inference_lora_with_reranker.py --cur_gpu_id 2 &
sleep 1s
CUDA_VISIBLE_DEVICES=3 nohup python3 inference_lora_with_reranker.py --cur_gpu_id 3 &
sleep 1s
CUDA_VISIBLE_DEVICES=4 nohup python3 inference_lora_with_reranker.py --cur_gpu_id 4 &
sleep 1s
CUDA_VISIBLE_DEVICES=5 nohup python3 inference_lora_with_reranker.py --cur_gpu_id 5 &
sleep 1s
CUDA_VISIBLE_DEVICES=6 nohup python3 inference_lora_with_reranker.py --cur_gpu_id 6 &
sleep 1s
CUDA_VISIBLE_DEVICES=7 nohup python3 inference_lora_with_reranker.py --cur_gpu_id 7 &
sleep 1s