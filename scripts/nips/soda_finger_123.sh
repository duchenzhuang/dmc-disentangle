# Scripts for SODA ENVs.


## Finger Spin on IIIS01
# SEED 0 on IIIS 01
## ORI-SODA
#CUDA_VISIBLE_DEVICES=0 python3 src/train.py \
#    --domain_name finger \
#    --task_name spin \
#	--algorithm soda \
#	--aux_lr 3e-4 \
#	--save_tb \
#	--seed 0 &
#
#
#sleep 5m
## INT-SODA
#CUDA_VISIBLE_DEVICES=1 python3 src/train.py \
#    --domain_name finger \
#    --task_name spin \
#	--algorithm soda \
#	--aux_lr 3e-4 \
#	--save_tb \
#	--use_intrinsic \
#	--in_gamma 10.0 \
#	--seed 0 &
#
#
#
#sleep 5m
#CUDA_VISIBLE_DEVICES=2 python3 src/train.py \
#    --domain_name finger \
#    --task_name spin \
#	--algorithm soda \
#	--aux_lr 3e-4 \
#	--save_tb \
#	--use_intrinsic \
#	--in_gamma 1.0 \
#	--seed 0 &
#
#
#
#sleep 5m
#CUDA_VISIBLE_DEVICES=3 python3 src/train.py \
#    --domain_name finger \
#    --task_name spin \
#	--algorithm soda \
#	--aux_lr 3e-4 \
#	--save_tb \
#	--use_intrinsic \
#	--in_gamma 0.1 \
#	--seed 0 &
#


# SEED 123 on IIIS 03
# ORI-SODA
CUDA_VISIBLE_DEVICES=1 python3 src/train.py \
    --domain_name finger \
    --task_name spin \
	--algorithm soda \
	--aux_lr 3e-4 \
	--save_tb \
	--seed 123 &

# INT-SODA
sleep 5m
CUDA_VISIBLE_DEVICES=3 python3 src/train.py \
    --domain_name finger \
    --task_name spin \
	--algorithm soda \
	--aux_lr 3e-4 \
	--save_tb \
	--use_intrinsic \
	--in_gamma 0.001 \
	--seed 123 &

sleep 5m
CUDA_VISIBLE_DEVICES=2 python3 src/train.py \
    --domain_name finger \
    --task_name spin \
	--algorithm soda \
	--aux_lr 3e-4 \
	--save_tb \
	--use_intrinsic \
	--in_gamma 0.01 \
	--seed 123 &

sleep 5m
CUDA_VISIBLE_DEVICES=0 python3 src/train.py \
    --domain_name finger \
    --task_name spin \
	--algorithm soda \
	--aux_lr 3e-4 \
	--save_tb \
	--use_intrinsic \
	--in_gamma 0.1 \
	--seed 123 &