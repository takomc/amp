# Automated Multi-level Preference for MLLMs
![](https://github.com/takomc/amp/blob/main/images/intro.png)
## Overview
We present an automated Multi-level Preference (AMP) framework for Reinforcement Learning from Human Feedback (RLHF), which generates the high-quality multi-level preference dataset without any human/AI annotators and employs multi-level DPO (MDPO) algorithm. Our AMP achieves SOTA performance across multiple hallucination benchmarks, including MMHal-Bench, MRHal-Bench, LLaVA-Bench, and POPE.

## Prepare
1. Install some important packages.
```Shell
conda create -n amp python=3.10 -y
conda activate amp
pip install --upgrade pip
pip install -r requirements.txt
```

2. Download Base Model

    [llava-7b-base](https://huggingface.co/mengxizhang/amp-7b-base)

    [llava-13b-base](https://huggingface.co/mengxizhang/amp-13b-base)

## Train
1. Prepare data from [[RLHF-V](https://rlhf-v.github.io/)], [[SILKIE](https://vlf-silkie.github.io/)], [[ShareGPT4V](https://sharegpt4v.github.io/)].

2. Download Data from [this link](https://pan.baidu.com/s/1-UnYpNZfMDtwp_T5emwG7g?pwd=6u3w).

3. Run the following code

```Shell
sh scripts/13b-v1.5/train_dpo.sh    # 13B
sh scripts/7b-v1.5/train_dpo.sh     # 7B
```

## Evaluation
### MMHal-Bench
1. Download data from [[MMHal-Bench](https://huggingface.co/datasets/Shengcao1006/MMHal-Bench)].
2. Run the script
```Shell
sh eval/eval_scripts/eval_mmhal.sh
```
### MRHal-Bench
1. Download data from [[MRHal-Bench](https://huggingface.co/datasets/mengxizhang/MRHal-Bench)].
2. Run the script
```Shell
sh eval/eval_scripts/eval_mrhal.sh
```

### LLaVA-Bench
1. Download data from [[LLaVA-Bench](https://github.com/llava-rlhf/LLaVA-RLHF/tree/main/Eval/llava)] and [[COCO](https://cocodataset.org/)] images.
2. Run the script
```Shell
sh eval/eval_scripts/eval_pope.sh
```

### POPE
1. Download data from [[POPE](https://github.com/AoiDragon/POPE/tree/e3e39262c85a6a83f26cf5094022a782cb0df58d/output/coco)] and [[COCO](https://cocodataset.org/)] images.
2. Run the script
```Shell
sh eval/eval_scripts/eval_llavab.sh
```

## Model Zoo
You can also use our trained models for evaluation. We provide the lora adpater of each version.
| Size| Dataset | Link | 
|----------|----------|-----------|
| 7B | MEG | [MEG-7B](https://huggingface.co/mengxizhang/amp_meg_7b) |
| 7B | IG | [IG-7B](https://huggingface.co/mengxizhang/amp_ig_7b) |
| 13B | MEG | [MEG-13B](https://huggingface.co/mengxizhang/amp_meg_13b) |
| 13B | IG | [IG-13B](https://huggingface.co/mengxizhang/amp_ig_13b) |


## Thanks
Our code is partly based on [[LLaVA](https://llava-vl.github.io/)], [[LLaVA-RLHF](https://llava-rlhf.github.io/)], and [[TRL](https://github.com/huggingface/trl)]. Thanks for their excllent work!
