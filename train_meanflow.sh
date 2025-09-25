#!/bin/bash
#SBATCH -o logs/job.%j.out
#SBATCH -p a10
#SBATCH --qos=qlong
#SBATCH -J tr_meanflow
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH --mem 32G

python meanse/train_meanflow.py \
    --train_tag meanflow \
    --train_name ncsnpp \
    --model_config /home/jiahe.wang/workspace/MeanSE/conf/meanflow/ncsnpp.yaml \
    --flow_ratio 0.5 \
    --max_interval 1.0 \
    --freeze_t_r_fuse False \
    --batch_size 1 \
    --num_gpu 1 \
    --learning_rate 1e-5 \
    --weight_decay 1e-6 \
    --train_set_path /home/jiahe.wang/workspace/DATA/data_vctk/train_noisy_16k \
    --valid_set_path /home/jiahe.wang/workspace/DATA/data_vctk/valid_noisy_16k \
    --fix_data_fs 16000 \
    --max_duration 40000 \
    --val_check_interval 100 \
    --init_from /home/jiahe.wang/workspace/MeanSE/exp/flow/ncsnpp/version_0/checkpoints/last.ckpt \