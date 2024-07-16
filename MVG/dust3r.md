# [DUSt3R](https://arxiv.org/abs/2312.14132)

_March_2024_

tl;dr: We cast the pairwise reconstruction problem as a regression of pointmaps, relaxing the hard constraints of usual projective camera models.

### Key ideas

每个子问题都解不好导致下一步引入噪声：传统方法将3D重建分解为一系列子问题，如匹配点(matching points)、寻找基本矩阵(finding essential matrices)、点的三角测量(triangulating points)等，每个步骤的都会为后续步骤引入噪声，增加整个流程的复杂性和工程难度。子问题间缺乏沟通：传统流程的各个子问题之间缺乏沟通。理想情况下，它们应该相互帮助，例如，密集重建应该从已经有的稀疏场景中获得帮助，反过来也是一样。关键步骤可能会失败：例如SfM（Structure-from-Motion）这一关键阶段在许多常见情况下容易失败，如场景视图数量较低、具有非Lambertian表面的对象、相机运动不足等情况。
