# multi-view geometry

[CMU 16385](https://www.cs.cmu.edu/~16385/s17/Slides/)

## Translation
Translation 具有2个DOF(Degree of Free，自由度)： $t_x, t_y$, 因此一个平移变换只需要一对对应点（correspondence）就能唯一确定。
不变量 平移变换不会改变物体的任何性质，因此平行、长度、面积、夹角等性质都不会改变。

## rotation
自由度 Rotation 只有1个DOF: $\theta$ ，
因此理论上来说也只需要一对点对即可求解，但是实际中一般很少出现纯旋转，而且点对之间总是存在噪声，因此很少有只使用一个点对求解旋转的。

不变量 旋转和平移类似，也不会改变物体的任何性质，因此平行、长度、面积、夹角等性质都不会改变。

## similarity 
是对刚体运动的一个扩展，在旋转矩阵上增加了一个全局 scale 系数
1.表达式
自由度 4个DOF：s $\theta, t_x, t_y$, 需要2对点对进行求解。




![image](https://github.com/Tony-Hou/Learning-AI/assets/26059901/ff936a49-7660-4c42-9cb6-c36348bc744d)
![image](https://github.com/Tony-Hou/Learning-AI/assets/26059901/3e17a0d5-530f-45f8-b7a0-0a4520b9efec)
