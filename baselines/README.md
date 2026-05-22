# Generative Representation Learning on Hyper-relational Knowledge Graphs via Masked Discrete Diffusion

This is the official code and data of [Generative Representation Learning on Hyper-relational Knowledge Graphs via Masked Discrete Diffusion]
### Accepted to the 43rd International Conference on Machine Learning (ICML 2026).

Baseline codes written by Seheon Kim (jacobpower@kaist.ac.kr).\
If you use this code, please cite our paper.

```bibtex
@inproceedings{krepe,
	author={Jaejun Lee, Seheon Kim, and Joyce Jiyoung Whang},
	title={Generative Representation Learning on Hyper-relational Knowledge Graphs via Masked Discrete Diffusion},
	booktitle={Proceedings of the 43rd International Conference on Machine Learning},
	year={2026},
	pages={}
}
```

# Baselines
We provide implementations of 7 baseline methods for fact generation, as detailed in our paper.\
For LLM-based baselines, we provide separate scripts for GPT-5.2 and Gemini 3.0 Pro.\
For all baselines, we provide separate scripts for each setting (*Scratch*, *Targeted*, and *Arbitrary Masking*).

# Prerequisites

**1. API Keys (For LLM Baselines)**

LLM-based methods require valid API credentials.

- OpenAI (GPT-5.2): Replace "API_KEY_HERE" in the respective scripts.
- Google (Gemini 3.0 Pro): Replace "API_KEY_HERE" and ensure the Google Application credentials file path is correctly set.

**2. MAYPL Training**

The following baselines rely on MAYPL:
- Iterative Prediction
- Gibbs Sampling
- Re-ranking

We provide the checkpoints of MAYPL.

To use the checkpoints:
1. Download and unzip `ckpt_MAYPL_for_KREPE.zip` file.
2. Place the unzipped `ckpt` folder in `./baselines/`.

You can download the checkpoints from [here](https://drive.google.com/file/d/1tJyo_LsfCDGQqQoT69S7fLi8zP_U3C5F/view?usp=sharing).

``maypl.py`` is the model implementation we used, which is provided in the current folder.

# Running the Baselines
Below are the commands used to produce the reported results for the *Arbitrary Masking* setting.

## Iterative Prediction

```bash
cd IterativePrediction
python IterativePrediction_arbi.py
```

## Gibbs Sampling

```bash
cd GibbsSampling
python GibbsSampling_arbi.py
```

## Re-ranking

```bash
cd Re-ranking
python re-ranking_gemini_arbi.py
```

## Neighbor Sets

```bash
cd NeighborSets
python NeighborSets_gemini_arbi.py
```

## Few-shot Facts

```bash
cd Few-shotFacts
python Few-shotFacts_gemini_arbi.py
```

## Random Facts

```bash
cd RandomFacts
python RandomFacts_gemini_arbi.py
```

## Autoregressive

```bash
cd Autoregressive
python Autoregressive_gemini_arbi.py
```

## License
Our codes are released under the CC BY-NC-SA 4.0 license.