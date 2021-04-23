# Scripts for Sparse ENVs.


# CartPole SwingUp
# ORI-SODA
CUDA_VISIBLE_DEVICES=0 python3 src/train.py \
    --domain_name cartpole \
    --task_name swingup_sparse \
	--algorithm soda \
	--aux_lr 3e-4 \
	--save_tb \
	--seed 0

# INT-SODA
CUDA_VISIBLE_DEVICES=0 python3 src/train.py \
    --domain_name cartpole \
    --task_name swingup_sparse \
	--algorithm soda \
	--aux_lr 3e-4 \
	--save_tb \
	--use_intrinsic \
	--in_gamma 1.0 \
	--seed 0

CUDA_VISIBLE_DEVICES=0 python3 src/train.py \
    --domain_name cartpole \
    --task_name swingup_sparse \
	--algorithm soda \
	--aux_lr 3e-4 \
	--save_tb \
	--use_intrinsic \
	--in_gamma 1.0 \
	--seed 0