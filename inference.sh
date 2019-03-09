#!/bin/bash
GPU_ID=0
# Phase2 test scoring
input_list="data/lists/phase2_list.txt"
for i in {1..3}
do
	config="data/opts/exp1_seed1/exp1_seed1_fold${i}_.opt"
	pth="data/opts/exp1_seed1/fold${i}/checkpoints/model_30.pth"
	output_list="data/opts/exp1_seed1/fold${i}/tta_epoch31_test_score.txt"
	CUDA_VISIBLE_DEVICES=$GPU_ID python inference.py --config $config --pth $pth --input_list $input_list --output_list $output_list --tta "hflip_5crop_rotate"
done 
for i in {1..3}
do
	config="data/opts/exp1_2stage_seed1/exp1_2stage_seed1_fold${i}_stage2.opt"
	pth="data/opts/exp1_2stage_seed1/fold${i}/checkpoints/model_30.pth"
	output_list="data/opts/exp1_2stage_seed1/fold${i}/tta_epoch31_test_score.txt"
	CUDA_VISIBLE_DEVICES=$GPU_ID python inference.py --config $config --pth $pth --input_list $input_list --output_list $output_list --tta "hflip_5crop_rotate"
done
for i in {1..3}
do
	config="data/opts/exp2_seed1/exp2_seed1_fold${i}_.opt"
	pth="data/opts/exp2_seed1/fold${i}/checkpoints/model_30.pth"
	output_list="data/opts/exp2_seed1/fold${i}/tta_epoch31_test_score.txt"
	CUDA_VISIBLE_DEVICES=$GPU_ID python inference.py --config $config --pth $pth --input_list $input_list --output_list $output_list --tta "hflip_5crop_rotate"
done 
for i in {1..3}
do
	config="data/opts/exp2_seed2/exp2_seed2_fold${i}_.opt"
	pth="data/opts/exp2_seed2/fold${i}/checkpoints/model_30.pth"
	output_list="data/opts/exp2_seed2/fold${i}/tta_epoch31_test_score.txt"
	CUDA_VISIBLE_DEVICES=$GPU_ID python inference.py --config $config --pth $pth --input_list $input_list --output_list $output_list --tta "hflip_5crop_rotate"
done 
for i in {1..3}
do
	config="data/opts/exp3b_seed1/exp3b_seed1_fold${i}_.opt"
	pth="data/opts/exp3b_seed1/fold${i}/checkpoints/model_30.pth"
	output_list="data/opts/exp3b_seed1/fold${i}/tta_epoch31_test_score.txt"
	CUDA_VISIBLE_DEVICES=$GPU_ID python inference.py --config $config --pth $pth --input_list $input_list --output_list $output_list --tta "hflip_5crop_rotate"
done
for i in {1..3}
do
	config="data/opts/exp3b_seed2/exp3b_seed2_fold${i}_.opt"
	pth="data/opts/exp3b_seed2/fold${i}/checkpoints/model_30.pth"
	output_list="data/opts/exp3b_seed2/fold${i}/tta_epoch31_test_score.txt"
	CUDA_VISIBLE_DEVICES=$GPU_ID python inference.py --config $config --pth $pth --input_list $input_list --output_list $output_list --tta "hflip_5crop_rotate"
done
for i in {1..3}
do
	config="data/opts/exp3c_seed1/exp3c_seed1_fold${i}_.opt"
	pth="data/opts/exp3c_seed1/fold${i}/checkpoints/model_30.pth"
	output_list="data/opts/exp3c_seed1/fold${i}/tta_epoch31_test_score.txt"
	CUDA_VISIBLE_DEVICES=$GPU_ID python inference.py --config $config --pth $pth --input_list $input_list --output_list $output_list --tta "hflip_5crop_rotate"
done
for i in {1..3}
do
	config="data/opts/exp3c_seed2/exp3c_seed2_fold${i}_.opt"
	pth="data/opts/exp3c_seed2/fold${i}/checkpoints/model_30.pth"
	output_list="data/opts/exp3c_seed2/fold${i}/tta_epoch31_test_score.txt"
	CUDA_VISIBLE_DEVICES=$GPU_ID python inference.py --config $config --pth $pth --input_list $input_list --output_list $output_list --tta "hflip_5crop_rotate"
done