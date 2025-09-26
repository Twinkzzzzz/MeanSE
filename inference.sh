python meanse/inference.py \
    --input_scp ./DATA/vctk/test_noisy_16k/wav.scp \
    --output_dir ./inference/test_meanflow \
    --ckpt_path ./exp/meanflow/ncsnpp/version_0/checkpoints/last.ckpt \
    --device cuda \
    --nfe 5