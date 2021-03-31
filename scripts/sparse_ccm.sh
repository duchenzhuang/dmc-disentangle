CUDA_VISIBLE_DEVICES=2 python3 src/train.py \
	--algorithm ours \
	--save_tb \
	--num_shared_layers 8 \
	--num_head_layers 3 \
	--seed 0