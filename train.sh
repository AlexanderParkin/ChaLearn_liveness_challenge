#!/bin/bash
GPU_ID="0,1,2,3"
# train exp1 seed1 on folds
for i in {1..3}
do
	config="data/opts/exp1_seed1/exp1_seed1_fold${i}_.opt"
	CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config $config
done
# train exp1_2stage stage1 seed1
for i in {1..3}
do
	config="data/opts/exp1_2stage_seed1/exp1_2stage_seed1_fold${i}_stage1.opt"
	CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config $config
done
#train exp1_2stage stage2 seed1
for i in {1..3}
do
	config="data/opts/exp1_2stage_seed1/exp1_2stage_seed1_fold${i}_stage2.opt"
	CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config $config
done
# train exp2 seed1 and seed2 on folds
for i in {1..3}
do
	config="data/opts/exp2_seed1/exp2_seed1_fold${i}_.opt"
	CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config $config
done
for i in {1..3}
do
	config="data/opts/exp2_seed2/exp2_seed2_fold${i}_.opt"
	CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config $config
done
# train exp3b seed1 and seed2 on folds
for i in {1..3}
do
	config="data/opts/exp3b_seed1/exp3b_seed1_fold${i}_.opt"
	CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config $config
done
for i in {1..3}
do
	config="data/opts/exp3b_seed2/exp3b_seed2_fold${i}_.opt"
	CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config $config
done
# train exp3c seed1 and seed2 on folds
for i in {1..3}
do
	config="data/opts/exp3c_seed1/exp3c_seed1_fold${i}_.opt"
	CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config $config
done
for i in {1..3}
do
	config="data/opts/exp3c_seed2/exp3c_seed2_fold${i}_.opt"
	CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --config $config
done