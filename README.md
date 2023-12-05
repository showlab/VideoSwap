# VideoSwap

**[VideoSwap: Customized Video Subject Swapping with Interactive Semantic Point Correspondence](https://arxiv.org/abs/2312.02087)**
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

[![arXiv](https://img.shields.io/badge/arXiv-2312.02087-b31b1b.svg)](https://arxiv.org/abs/2312.02087)
[![Project Page](https://img.shields.io/badge/Project-Website-orange)](https://videoswap.github.io/)



https://github.com/showlab/VideoSwap/assets/31696690/7e9395cd-71bb-4d06-960f-5151a5dd06fb



**VideoSwap** is a framework that supports swapping users' _**customized concepts**_ into videos while _**preserving the background**_. 

>Current diffusion-based video editing primarily focuses on structure-preserved editing by utilizing various dense correspondences to ensure temporal consistency and motion alignment. However, these approaches are often ineffective when the target edit involves a shape change.
To embark on video editing with shape change, we explore customized video subject swapping in this work, where we aim to replace the main subject in a source video with a target subject having a distinct identity and potentially different shape.
In contrast to previous methods that rely on dense correspondences, we introduce the VideoSwap framework that exploits semantic point correspondences, inspired by our observation that only a small number of semantic points are necessary to align the subject's motion trajectory and modify its shape. We also introduce various user-point interactions (\eg, removing points and dragging points) to address various semantic point correspondence. Extensive experiments demonstrate state-of-the-art video subject swapping results across a variety of real-world videos.

For more see the [project webpage](https://videoswap.github.io/).

## 📜 License and Acknowledgement

This codebase builds on [diffusers](https://github.com/huggingface/diffusers). Thanks for open-sourcing! Besides, we acknowledge following amazing open-sourcing projects:

- AnimateDiff (https://github.com/guoyww/AnimateDiff).


- Mix-of-Show (https://github.com/TencentARC/Mix-of-Show).


- Co-tracker (https://github.com/facebookresearch/co-tracker).

## Citation

```bibtex

@article{gu2023videoswap,
  title={VideoSwap: Customized Video Subject Swapping with Interactive Semantic Point Correspondence},
  author={Gu, Yuchao and Zhou, Yipin and Wu, Bichen and Yu, Licheng and Liu, Jia-Wei and Zhao, Rui and Wu, Jay Zhangjie and Zhang, David Junhao and Shou, Mike Zheng and Tang, Kevin},
  journal={arXiv preprint arXiv:2312.02087},
  year={2023}
}

```
