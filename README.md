# MeanSE: Efficient Generative Speech Enhancement with Mean Flows

This is the official implementation of ["MeanSE: Efficient Generative Speech Enhancement with Mean Flows"](https://arxiv.org/abs/2509.21214) submitted to *ICASSP 2026*.

>**Abstract:**<br>
Speech enhancement (SE) improves degraded speech's quality, with generative models like flow matching gaining attention for their outstanding perceptual quality. However, the flow-based model requires multiple numbers of function evaluations (NFEs) to achieve stable and satisfactory performance, leading to high computational load and poor 1-NFE performance. In this paper, we propose MeanSE, an efficient generative speech enhancement model using mean flows, which models the average velocity field to achieve high-quality 1-NFE enhancement. Experimental results demonstrate that our proposed MeanSE significantly outperforms the flow matching baseline with a single NFE, exhibiting extremely better out-of-domain generalization capabilities.

---

## Quick Start

```
git clone https://github.com/Twinkzzzzz/MeanSE.git
cd MeanSE
pip install espnet
pip install -e ./
```

## Training

We have provided training bash script examples in `meanse/train_flow.sh` and `meanse/train_meanflow.sh`. Some key arguments are illustrated as follows:

* `--model_config` specifies the `.yaml` configuration setting file of the DNN model. We have provieded examples in `conf/flow/ncsnpp.yaml` and `conf/meanflow/ncsnpp.yaml`.

* `--train_set_path` and `--valid_set_path` specify the used dataset, following the dataset usage in [Espnet](https://github.com/espnet/espnet).

* `--flow_ratio` defines the flow ratio in training MeanSE, as explained in *Section 2.2.2* in the paper.

* `--max_interval` specifies the maximum value of the sampling time interval during training MeanSE, as explained in *Section 2.2.2* in the paper.

* `--init_from` specifies the checkpoint used for model initialization.

## Inference

We have provided inference bash script examples in `meanse/inference.sh`. Some key arguments are illustrated as follows:

* `--input_scp` specifies the testing data to be inferenced, in the form of `.scp` file.

* `--ckpt_path` appoints the checkpoint to be tested.

* `--output_dir` specifies the output directory, where a `.scp` file and a `wav` folder will be created.

* `--force_model_type` defines whether the appointed checkpoint is a `flow` model or a `meanflow` model. It can be ignored since `meanse/inference.py` automatically detects the model type of the checkpoint. We provide this argument for special needs.

* `--nfe` specifies the number of function evaluations in the inference stage.

---
*MeanSE is developed based on the open-source code framework provided by [Urgent 2026 Challenge](https://github.com/urgent-challenge/urgent2026_challenge_track1)*