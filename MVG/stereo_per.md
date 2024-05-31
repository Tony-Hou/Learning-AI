# Stereo peception
## Datasets
###

| Datasets | link    |   other   |
| -------  | :-----: |  :------: |
| Middlebury| [链接](https://vision.middlebury.edu/stereo/eval3/)    |  |
| KITTI      |    |      |
|  ETH3D     |    |      |

## Metrics

| Datasets  | 评估区域  | 评估指标  |        |
| -------   | :------: | :-----: | :----: |  
| Middlebury|  <li> dics(Depth Discontinuity Region) 视差不连续区域 <li> all(All Region) 全部区域 <li> non-occ (Non-occlusion region) 非遮挡区域   |       [reference](https://blog.csdn.net/RadiantJeral/article/details/85172432) |
|           |          |   |

## 学习资料
| |  |  |
| | [计算机摄影学](https://zhuanlan.zhihu.com/p/482877544) |  |
| |  |  |


### TI
TDA4VM 
深度和运动处理加速器 (DMPAC) 包含两个模块，分别是加速立体深度估算的立体视差引擎 (SDE)
和加速 DOF 的密集光流引擎（DOF 引擎）。
- 双目深度加速器以 80 M/sec 的速度产生 192 个视差
- 视觉处理加速器支持高达 180 度的双目矫正和高达 720 MP/sec 的处理速度
- SDE 的最大视差搜索范围为191 (0 ~ 191)。 其实际检测范围将取决于焦距和基线距离。
对于 Kitti 2012和 Kitti 2015数据集、SDE 的3像素误差约为7.3%~ 7.8%。

立体视差性能对左右摄像头的图像均匀性非常敏感、因此最好使左右摄像头在每帧使用相同的 WB 增益/曝光设置。[AEB/AE](https://blog.csdn.net/Cmatrix204/article/details/126410710)
TDA4 ISP 是否支持此类 AWB/AE 主从功能？ 比如、让左摄像头 视频管道成为主摄像头、它计算每帧的 WB 增益和曝光设置、右摄像头视频管道同步 WB 增益/曝光 、左摄像头计算的是什么？
[TDA4VM](https://www.ti.com.cn/cn/lit/an/zhcaax5/zhcaax5.pdf?ts=1717145818312&ref_url=https%253A%252F%252Fwww.google.com%252F)
