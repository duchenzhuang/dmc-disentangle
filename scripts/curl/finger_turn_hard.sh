CUDA_VISIBLE_DEVICES=3 python3 src/train.py \
	--domain_name finger \
  --task_name turn_hard \
	--algorithm curl \
	--seed 0 \
	--train_steps 1000k \
	--aux_update_freq 1 \
	--capacity 200k \
	--num_shared_layers 4 \
	--save_tb
#	--in_gamma 0.1 \
#	--in_decay 0.8 \
#	--use_intrinsic