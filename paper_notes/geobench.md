# GeoBench: Benchmarking and Analyzing Monocular Geometry Estimation Models

tl;dr
提出了用于评估SOTA判别式和生成深度和表面法线估计基础模型的综合单目几何基准

1.在相同的训练配置下，用大数据预训练的判别模型（如DINOv2）可以优于用小规模高质量合成数据的稳定扩散预训练的生成模型。

2.合成数据对于细粒度深度估计至关重要。数据质量是比模型体系结构和数据规模更重要的因素。

3.归纳偏差是表面法线估计的关键。 作者：3D视觉工坊 https://www.bilibili.com/read/cv37123029/?share_source=copy_link&share_medium=iphone&bbid=9340a68bcfd79446e9f1737aec9147a0&ts=1729130169 出处：bilibili
