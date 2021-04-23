#CUDA_VISIBLE_DEVICES=0 python3 src/train.py \
#    --domain_name walker \
#    --task_name run \
#	--algorithm soda \
#	--aux_lr 3e-4 \
#	--save_tb \
#	--seed 0  &

CUDA_VISIBLE_DEVICES=3 python3 src/train.py \
    --domain_name cartpole \
    --task_name swingup_sparse \
	--algorithm soda \
	--aux_lr 3e-4 \
	--save_tb \
	--use_intrinsic \
	--seed 0  &

CUDA_VISIBLE_DEVICES=1 python3 src/train.py \
    --domain_name cartpole \
    --task_name swingup_sparse \
	--algorithm soda \
	--aux_lr 3e-4 \
	--save_tb \
	--seed 0 &


#CUDA_VISIBLE_DEVICES=2 python3 src/train.py \
#    --domain_name ball_in_cup \
#    --task_name catch \
#	--algorithm soda \
#	--aux_lr 3e-4 \
#	--save_tb \
#	--seed 0  &
#
#CUDA_VISIBLE_DEVICES=3 python3 src/train.py \
#    --domain_name finger \
#    --task_name spin \
#	--algorithm soda \
#	--aux_lr 3e-4 \
#	--save_tb \
#	--seed 0  &