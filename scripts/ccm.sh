CUDA_VISIBLE_DEVICES=3 python3 src/train.py \
	--algorithm ccm \
	--projection_dim 2048 \
	--aux_update_freq 1 \
	--seed 0