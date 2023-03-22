#!/bin/bash
#SBATCH -J ov-seg_demo_test
#SBATCH --comment "ov-seg demo test"
#SBATCH -p gpu
#SBATCH --qos level0
#SBATCH -N 1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=1
#SBATCH -t 00-01:00:00
#SBATCH --account research

#SBATCH -o "/scratch/lipuhao/dev/MAH/DataPreprocess/ov-seg/outputs/slurm/%x.%j.out"
#SBATCH -e "/scratch/lipuhao/dev/MAH/DataPreprocess/ov-seg/outputs/slurm/%x.%j.err"

python demo.py --bs_test 16 --config-file configs/ovseg_swinB_vitL_demo.yaml --class-names 'Oculus' 'Ukulele'  --input ./resources/demo_samples/sample_03.jpeg --output ./pred --opts MODEL.WEIGHTS ./ckpts/ovseg_swinbase_vitL14_ft_mpt.pth

