[3D Point Position Embedding, 3DPPE](https://arxiv.org/abs/2211.14710)
Nov 2022
tl;dr: 
该方案旨在解决当前基于Transformer范式的环视3D障碍物检测中存在的图像与锚点位置编码不一致以及沿射线方向的误检导致后处理逻辑复杂等问题，在获得卓越性能的同时进一步降低了后处理的复杂度，
同比petr-v1/2以及streampetr均取得显著提升

# Overall impression

# Key ideas
由于3DPPE在构建图像特征的位置编码时引入了显式的深度信息，使得对应的位置先验与真实物理世界的分布更为一致，从而有效的减缓了沿射线方向的误检
针对环视3D目标检测问题，基于优秀的PETR框架(说实话，petr以及streampetr是真好)，分析了现有3D目标检测中位置编码的诸多区别，提出了一种新的3D点编码(图1.b)，解决了之前射线编码(图1.a)缺乏细粒位置先验的问题，统一了图像特征和query的位置表征，在环视3D检测上实现了卓越性能。在最先进的petr v1/v2以及streampetr上都得到了提升，我们罗列了在经典的v2-99主干+800x320的分辨率下的效果(这个配置里的主干网在深度数据集上预训练过、并且小分辨率对训练的要求降低从而更亲民，适合做消融实验对比性能)
<img width="500" alt="image" src="https://github.com/Tony-Hou/Learning-AI/assets/26059901/d744507e-0d4b-4823-9f61-09f734483d60">

## Reference
[3dppe](https://mp.weixin.qq.com/s?__biz=MzU2NjU3OTc5NA==&mid=2247576915&idx=1&sn=4e6d73c3bbeb330f50f62a2825773c72&chksm=fca9aa6ecbde2378678046b0d30481148f6832892ee1d488bcc4c17ead5af6f141f245add017&scene=27)
