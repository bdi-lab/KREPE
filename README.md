# Generative Representation Learning on Hyper-relational Knowledge Graphs via Masked Discrete Diffusion

This is the official code and data of [Generative Representation Learning on Hyper-relational Knowledge Graphs via Masked Discrete Diffusion]
### Accepted to the 43rd International Conference on Machine Learning (ICML 2026).

Codes written by Jaejun Lee (jjlee98@kaist.ac.kr) and Seheon Kim (jacobpower@kaist.ac.kr).\
If you use this code, please cite our paper.
<!--
```bibtex
@inproceedings{krepe,
	author={Jaejun Lee, Seheon Kim, and Joyce Jiyoung Whang},
	title={Generative Representation Learning on Hyper-relational Knowledge Graphs via Masked Discrete Diffusion},
	booktitle={Proceedings of the 43rd International Conference on Machine Learning},
	year={2026},
	pages={}
}
```
-->
```bibtex
@inproceedings{krepe,
  author={Jaejun Lee and Seheon Kim and Joyce Jiyoung Whang},
  title={Generative Representation Learning on Hyper-relational Knowledge Graphs via Masked Discrete Diffusion},
  year={2026},
	journal={arXiv preprint arXiv:2605.24064},
	doi={10.48550/arXiv.2605.24064},
}
```


## Requirements

We used python 3.11.13 and PyTorch 2.6.0 with cudatoolkit 11.8.

You can install all requirements (except python) with:

```setup
pip install -r requirements.txt
```

## Baselines
For baseline instructions, please refer to ``./baselines/README.md``.

## Training
We used an NVIDIA RTX A6000 for all datasets.

## Reproducing Reported Results using Checkpoints

We provide the checkpoints used to produce all reported results.

To use the checkpoints:
1. Download and unzip `ckpt_KREPE.zip` file.
2. Place the unzipped `ckpt` folder in the same directory with the codes.

You can download the checkpoints from [here](https://drive.google.com/file/d/1hMiXpNkMUKcz8G2glJ2aUHv0Y4RjrxOI/view?usp=sharing).


### wd50k (Training on train+valid sets)

The command to train KREPE used to produce the results in our paper:

```python
python train.py --dataset_name wd50k-eval --exp ICML2026 --log_name krepe --dim 128 --act GELU --start_epoch 0 --num_epoch 1950 --val_dur 1950 --val_size 4096 --num_layer 16 --num_head_ent 4 --num_head_rel 4 --early_stop 0  --batch_size 2048 --model_dropout 0.1 --weight_decay 0.01 --grad_clip 1.0 --lr_max 0.001 --lr_min 1e-05 --train_graph_ratio 0.7 --warmup_epoch 200
```

### WikiPeople- (Training on train+valid sets)

The command to train KREPE used to produce the results in our paper:

```python
python train.py --dataset_name WikiPeople--eval --exp ICML2026 --log_name krepe --dim 256 --act GELU --start_epoch 0 --num_epoch 2000 --val_dur 2000 --val_size 4096 --num_layer 12 --num_head_ent 8 --num_head_rel 32 --early_stop 0  --batch_size 4096 --model_dropout 0.1 --weight_decay 0.1 --grad_clip 1.0 --lr_max 0.0004 --lr_min 1e-05 --train_graph_ratio 0.7 --warmup_epoch 200 --mask_eq_init --wd_mlp_ln
```

### WikiPeople (Training on train set)

The command to train KREPE used to produce the results in our paper:

```python
python train.py --dataset_name WikiPeople --exp ICML2026 --log_name krepe --dim 256 --act GELU --start_epoch 0 --num_epoch 1750 --val_dur 1750 --val_size 4096 --num_layer 12 --num_head_ent 8 --num_head_rel 32 --early_stop 0  --batch_size 4096 --model_dropout 0.1 --weight_decay 0.1 --grad_clip 1.0 --lr_max 0.0004 --lr_min 1e-05 --train_graph_ratio 0.7 --warmup_epoch 200 --mask_eq_init --wd_mlp_ln
```

## Link Prediction

The command to evaluate KREPE used to produce the results for WikiPeople in our paper:
```python
python test.py --dataset_name WikiPeople --exp ICML2026 --log_name krepe --test_epoch 1750 --dim 256 --act GELU --num_layer 12 --num_head_ent 8 --num_head_rel 32 --mask_eq_init
```

## Fact Generation

The command to generate facts of length 5 for *Scratch* setting using KREPE on WD50K:

```python
python sampling_scratch.py --dataset_name wd50k-eval --exp ICML2026 --log_name krepe --test_epoch 1950 --fact_len 5 --fact_num 100 --steps 1000 --ent_p 0.15 --ent_temp 1 --rel_p 0.05 --rel_temp 1 --dim 128 --act GELU --num_layer 16 --num_head_ent 4 --num_head_rel 4
```

The command to generate facts for *Arbitrary Masking* setting using KREPE on WD50K:

```python
python sampling_mask.py --dataset_name wd50k-eval --exp ICML2026 --log_name krepe --test_epoch 1950 --input_file ./data/wd50k-eval/arbitrary_masking.txt --steps 1000 --ent_p 0.7 --ent_temp 0.55 --rel_p 0.5 --rel_temp 0.5 --max_retries 10 --dim 128 --act GELU --num_layer 16 --num_head_ent 4 --num_head_rel 4
```

For the sampling hyperparameters used in other settings and datasets, please refer to our paper.

The generated facts can be evaluated by running ``verify.sh`` under ``./verification``.\
Evaluation requires an OpenAI API key. Please replace ``"API_KEY_HERE"`` in ``verification/gpt_verify.py`` with your actual key.

## License
Our codes are released under the CC BY-NC-SA 4.0 license.
