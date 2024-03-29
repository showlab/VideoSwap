# VideoSwap

**[CVPR 2024] - [VideoSwap: Customized Video Subject Swapping with Interactive Semantic Point Correspondence](https://arxiv.org/abs/2312.02087)**
<br/>

[Yuchao Gu](https://ycgu.site/),
[Yipin Zhou](https://yipin.github.io/),
[Bichen Wu](https://scholar.google.com/citations?user=K3QJPdMAAAAJ&hl=en),
[Licheng Yu](https://lichengunc.github.io/),
[Jia-Wei Liu](https://jia-wei-liu.github.io/),
[Rui Zhao](https://ruizhaocv.github.io/),
[Jay Zhangjie Wu](https://zhangjiewu.github.io/),<br/> 
[David Junhao Zhang](https://junhaozhang98.github.io/),
[Mike Zheng Shou](https://sites.google.com/view/showlab),
[Kevin Tang](https://ai.stanford.edu/~kdtang/)
<br/>

Showlab, National University of Singapore; GenAI, Meta

[![arXiv](https://img.shields.io/badge/arXiv-2312.02087-b31b1b.svg)](https://arxiv.org/abs/2312.02087)
[![Project Page](https://img.shields.io/badge/Project-Website-orange)](https://videoswap.github.io/)

https://github.com/showlab/VideoSwap/assets/31696690/7e9395cd-71bb-4d06-960f-5151a5dd06fb

**VideoSwap** is a framework that supports swapping users' _**customized concepts**_ into videos while _**preserving the background**_. 

>Current diffusion-based video editing primarily focuses on structure-preserved editing by utilizing various dense correspondences to ensure temporal consistency and motion alignment. However, these approaches are often ineffective when the target edit involves a shape change.
To embark on video editing with shape change, we explore customized video subject swapping in this work, where we aim to replace the main subject in a source video with a target subject having a distinct identity and potentially different shape.
In contrast to previous methods that rely on dense correspondences, we introduce the VideoSwap framework that exploits semantic point correspondences, inspired by our observation that only a small number of semantic points are necessary to align the subject's motion trajectory and modify its shape. We also introduce various user-point interactions (\eg, removing points and dragging points) to address various semantic point correspondence. Extensive experiments demonstrate state-of-the-art video subject swapping results across a variety of real-world videos.

For more see the [project webpage](https://videoswap.github.io/).

## 🛑 Disclaimer

This repository is a re-implementation of VideoSwap conducted by the first author during his time at NUS. The goal of this repository is to replicate the original paper's findings and results, primarily for academic and research purposes. **Due to legal considerations, the code is available as open-source on a per-request basis. Please fill in [this form](https://forms.gle/TB9fatZEzXTUwoj88) to request code access.**

## 🚩 Updates

- [ ] Interative gradio demo.
- [ ] Guidance to train on users' own data.
- [x] **[Mar. 28, 2024.]** Training Code Released (to reproduce all results in paper).
- [x] **[Mar. 28, 2024.]** Inference Code Released (to reproduce all results in paper).

## 🔧 Dependencies and Installation
The code verified under an A100 GPU (require 12 GB for train and inference):
- Python == 3.10
- Pytorch == 2.2.1
- Diffusers == 0.19.3

Please follow the instruction to create the environment:

```bash
# Setup Conda Environment
conda create -n videoswap python=3.10
conda activate videoswap

# Install the pytorch/xformer
conda install pytorch==2.2.1 torchvision==0.17.1 torchaudio==2.2.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install xformers -c xformers/label/dev

# Install other dependences
pip install -r requirements.txt
```

## Quick Start

To reimplement the paper results, we provide all preprocessed datasets and checkpoints:

```bash
# install gdown
pip install gdown
apt install git-lfs

# automatically download dataset, pretrained models and our results
bash scripts/prepare_dataset_model.sh
```

To inference our pretrained models:

```bash
# choose one config in options/test_videoswap, for example
python test.py -opt options/test_videoswap/animal/2001_catheadturn_T05_Iter100/2001_catheadturn_T05_Iter100.yml
```


To train the model by yourself based on provided data:
```bash
# choose one config in options/train_videoswap, for example
python train.py -opt options/train_videoswap/animal/2001_catheadturn_T05_Iter100/2001_catheadturn_T05_Iter100.yml
```

## 📜 License and Acknowledgement

This codebase builds on [diffusers](https://github.com/huggingface/diffusers). Thanks for open-sourcing! Besides, we acknowledge following amazing open-sourcing projects:

- AnimateDiff (https://github.com/guoyww/AnimateDiff).


- Mix-of-Show (https://github.com/TencentARC/Mix-of-Show).


- Co-tracker (https://github.com/facebookresearch/co-tracker).

## Citation

If you find this repository useful in your work, consider citing the following papers and giving a ⭐ to the public repository (https://github.com/showlab/VideoSwap) to allow more people to discover this repo:

```bibtex
@article{gu2023videoswap,
  title={Videoswap: Customized video subject swapping with interactive semantic point correspondence},
  author={Gu, Yuchao and Zhou, Yipin and Wu, Bichen and Yu, Licheng and Liu, Jia-Wei and Zhao, Rui and Wu, Jay Zhangjie and Zhang, David Junhao and Shou, Mike Zheng and Tang, Kevin},
  journal={arXiv preprint arXiv:2312.02087},
  year={2023}
}

@article{gu2024mix,
  title={Mix-of-show: Decentralized low-rank adaptation for multi-concept customization of diffusion models},
  author={Gu, Yuchao and Wang, Xintao and Wu, Jay Zhangjie and Shi, Yujun and Chen, Yunpeng and Fan, Zihan and Xiao, Wuyou and Zhao, Rui and Chang, Shuning and Wu, Weijia and others},
  journal={Advances in Neural Information Processing Systems},
  volume={36},
  year={2024}
}

@inproceedings{wu2023tune,
  title={Tune-a-video: One-shot tuning of image diffusion models for text-to-video generation},
  author={Wu, Jay Zhangjie and Ge, Yixiao and Wang, Xintao and Lei, Stan Weixian and Gu, Yuchao and Shi, Yufei and Hsu, Wynne and Shan, Ying and Qie, Xiaohu and Shou, Mike Zheng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={7623--7633},
  year={2023}
}
```