# Representation Alignment for Generation: training diffusion transformers is easier than you think

提出了一个新技术: REPresentation Alignment(REPA),简单来说,就是让模型更聪明地利用表征来生成图片。
根据他们的研究,使用预训练的视觉编码器(例如DINOV2)的高质量表征,可以大幅提升扩散模型的性能。
这项技术能让训练速度加快17.5倍!想象一下,那些美轮美奂的生成图片背后的秘密,竟然是这个小小的对齐技巧!
当然,这项研究也得到了AI大牛Yann LeCun的认
可。他指出,即便是只对生成像素感兴趣,也应该考
虑特征预测损失。这种方法能帮助解码器内部表征更
好地预测特征。
他们还发现,扩散模型的隐藏状态可以与自监督视觉
表征对齐,从而提高生成质量。这种正则化技术不仅
简单,还非常有效!
![image](https://github.com/user-attachments/assets/6cafe46c-961d-4a3a-a48c-e9524bb85146)


# 启发
最近也在思考生成仿真数据时，对于时序上存在的一致性
