# Freespace

## grid map
[grid map code](https://github.com/ANYbotics/grid_map)

### gridmap
- 地图文件 .pgm 和与之匹配的yaml文件
```yaml
image: fishbot_map.pgm
mode: trinary
resolution: 0.05
origin: [-3.37, -2.88, 0]
negate: 0
occupied_thresh: 0.65
free_thresh: 0.25
```

image：图像名称
mode：图像模式，默认为trinary(三进制)，还有另外两个可选项scale(缩放的)和raw(原本的值)。
resolution：分辨率，一个栅格对应的物理尺寸，单位为m。0.05则表示一个栅格为0.05m
origin：地图原点位置，单位是m。
negate：是否反转图像
cooupied_thresh：占据阈值
free_thresh：空闲阈值


### costmap_2d 
由于占据栅格地图通常是单层结构，更适用于简单的路径规划功能，当机器人或无人车对于环境信息精确度要求不高时可以利用此地图进行规划，当规划模块对精度与安全性要求较高时，更好的选择是利用代价地图（CostMap）进行规划
Costmap是无人车或者机器人融合多传感器信息后建立维护的二维或三维栅格地图，在地图中障碍物信息会被转换到对应的栅格中并且利用相应的规则对障碍物进行膨胀。ROS的官方Navigation库中最早利用costmap进行障碍物的位置判断以及在此基础上利用DWA（原理可参见我之前的系列博客）算法进行了局部规划，后来在日本的开源框架Autoware中也基于costmap实现了乘用车的局部规划功能（HybridAstar算法）

costmap ref:
[costmap_2d](http://wiki.ros.org/costmap_2d/)

grid map ref:
[grid map ](https://en.wikipedia.org/wiki/Occupancy_grid_mapping#cite_note-1)
[]()
[gridmaps](http://ais.informatik.uni-freiburg.de/teaching/ws13/mapping/pdf/slam10-gridmaps.pdf)
[probabilistics robotics](https://github.com/yvonshong/Probabilistic-Robotics.git)

### Ref
[development of free space detection by sensor fusion ](https://www.jstage.jst.go.jp/article/jsaeronbun/51/2/51_20204100/_pdf)
[High Resolution Radar-based Occupancy Grid Mapping and Free
Space Detection ](https://www.scitepress.org/PublishedPapers/2018/66673/66673.pdf)
[Freespace detection and examination based on surround occupancy grid](https://www.eaiib.agh.edu.pl/wp-content/uploads/2023/10/PhD_thesis_Marek_Szlachetka.pdf)
[Robust free space detection in occupancy grid maps by methods of image analysis and dynamic B-spline contour tracking](https://ieeexplore.ieee.org/document/6338636)
