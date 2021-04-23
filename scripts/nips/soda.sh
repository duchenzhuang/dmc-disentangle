# ORI-SODA

CUDA_VISIBLE_DEVICES=0 python3 src/train.py \
    --domain_name finger \
    --task_name spin \
	--algorithm soda \
	--aux_lr 3e-4 \
	--save_tb \
	--use_intrinsic \
	--in_gamma 1.0 \
	--seed 0

#CUDA_VISIBLE_DEVICES=1 python3 src/train.py \
#    --domain_name finger \
#    --task_name spin \
#	--algorithm soda \
#	--aux_lr 3e-4 \
#	--save_tb \
#	--use_intrinsic \
#	--in_gamma 0.1 \
#	--seed 0 &
#
#CUDA_VISIBLE_DEVICES=1 python3 src/train.py \
#    --domain_name cartpole \
#    --task_name swingup_sparse \
#	--algorithm soda \
#	--aux_lr 3e-4 \
#	--save_tb \
#	--seed 0 &


#CUDA_VISIBLE_DEVICES=2 python3 src/train.py \
#    --domain_name ball_in_cup \
#    --task_name catch_sparse \
#	--algorithm soda \
#	--aux_lr 3e-4 \
#	--save_tb \
#	--seed 0  &
#
#CUDA_VISIBLE_DEVICES=3 python3 src/train.py \
#    --domain_name cartpole \
#    --task_name swingup_sparse \
#	--algorithm soda \
#	--aux_lr 3e-4 \
#	--save_tb \
#	--seed 0 &


# INTRINSIC SODA
#CUDA_VISIBLE_DEVICES=3 python3 src/train.py \
#    --domain_name cartpole \
#    --task_name swingup_sparse \
#	--algorithm soda \
#	--aux_lr 3e-4 \
#	--save_tb \
#	--use_intrinsic \
#	--seed 0  &

#CUDA_VISIBLE_DEVICES=1 python3 src/train.py \
#    --domain_name cartpole \
#    --task_name swingup_sparse \
#	--algorithm soda \
#	--aux_lr 3e-4 \
#	--save_tb \
#	--use_intrinsic \
#	--in_gamma 1.0 \
#	--seed 0 &
#
#CUDA_VISIBLE_DEVICES=1 python3 src/train.py \
#    --domain_name cartpole \
#    --task_name swingup_sparse \
#	--algorithm soda \
#	--aux_lr 3e-4 \
#	--save_tb \
#	--use_intrinsic \
#	--in_gamma 0.1 \
#	--seed 0 &
#
#CUDA_VISIBLE_DEVICES=3 python3 src/train.py \
#    --domain_name cartpole \
#    --task_name swingup_sparse \
#	--algorithm soda \
#	--aux_lr 3e-4 \
#	--save_tb \
#	--use_intrinsic \
#	--in_gamma 0.01 \
#	--seed 0 &


# 好像没有finger spin_sparse这个东西