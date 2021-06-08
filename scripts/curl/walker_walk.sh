CUDA_VISIBLE_DEVICES=2 python3 src/train.py \
	--domain_name walker \
  --task_name walk \
	--algorithm curl \
	--seed 0 \
	--aux_update_freq 1 \
	--capacity 200k \
	--num_shared_layers 4 \
	--save_tb
#	--in_gamma 0.05 \
#	--in_decay 0.8 \
#	--use_intrinsic