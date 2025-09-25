python meanse/inference.py \
    --input_scp /home/jiahe.wang/workspace/DATA/data_vctk/test_noisy_16k/wav.scp \
    --output_dir ./inference/test_meanflow \
    --ckpt_path /home/jiahe.wang/workspace/MeanSE/exp/meanflow/ncsnpp/version_0/checkpoints/last.ckpt \
    --device cuda \
    --nfe 5