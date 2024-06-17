# 2D occupancy grid mapping 中怎么标记不规则物体和悬空物体的

### 语义分割存在的问题
- freespace 边界预测不稳定，怎么解决这个问题，因为大多语义分割是对像素进行的分类，没有boundaries loss得监督，解决方法 [focus the entire loss function on the boundaries instead of weighting all pixel patches](file:///Users/linjie.hou/Documents/%E5%A4%9A%E8%A7%86%E5%9B%BE%E5%87%A0%E4%BD%95/stereo/freespace_googlenet.pdf)
- 

边界怎么进行跟踪融合的


## 2D 空间中的解决方法

边缘是高频信号

[canny: A Computational Approach to Edge Detection](https://ieeexplore.ieee.org/document/4767851/metrics#metrics)

# 3D occupancy grid 时序帧的occupancy 怎么更新的，静态和动态的物体怎么更新的
Occupancy Network学术上讲就是建模物体详细几何和语义的通用且连贯的表征

一个需要从输入图像中联合估计场景中每个voxel的占据状态和语义标签的模型，其中占据状态分为free，occupied和unobserved三种，对于occupied的voxel，
还需要分配其语义标签。而对于没有标注的物体类别，统一划分为General Objects(GOs)，GOs少见但为了安全起见是必须的，否则检测时经常检测不到。
Occupancy Network理论上能解决无法识别物体的难题，但实际中不能。很简单，Occupancy Network是一种预测性质的神经网络，它不可能达到100%的准确度，
自然也就有漏网之鱼，还是有无法识别的物体无法探测

## 3D 空间中的解决方法是预测flow,

### 参考资料
[Indoor Segmentation and Support Inference from RGBD Images](Indoor Segmentation and Support Inference from RGBD Images)
[2018 bosch](Occupancy Networks: Learning 3D Reconstruction in Function Space)
[HOPE](https://mp.weixin.qq.com/s/YwsFzwjmZ8hvlTnmb0jw3g)
[Occupancy flow](Occupancy flow fields for motion forecasting in autonomous driving)
[]()
