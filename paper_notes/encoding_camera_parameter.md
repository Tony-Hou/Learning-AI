# camera parameter
摄像头参数包括内外参，通常使用MLP进行编码，为什么使用MLP进行编码
## bevdepth
### bevdepth 算法流程
1. 利用Resnet作为backbone提取环视六张图特征, second_fpn作为neck进行特征融合。
2. 利用深度估计网络DepthNet提取环视图像特征的context和depth信息：
  ①. 由于深度估计依赖于相机参数，因此利用mlp将相机参数维度由[6, 27]—>[6, 512, 1, 1]，作为SE注意力权重。
  ②. 使其分别与图像特征进行注意力操作，并利用1*1卷积，进行通道缩放，获得context特征与depth特征
3. 对depth特征进行softmax操作获得深度估计分布
4. 和LSS思路一样，Context Feature和Depth Distribution作外积获得图像语义特征和深度特征的综合信息。
5. GetGeometry()构建视锥点云，并利用相机参数将视锥点云转换到车辆坐标系下的空间位置。
6. 将上述4和5的输出作为输入，根据视锥点云在ego周围空间的位置索引，把点云特征分配到BEV pillar中，然后对同一个pillar中的点云特征进行sum-pooling处理，输出[b, 80, 128, 128]的BEV特征。
7. BEVDepth采用了多帧融合， 其中当前帧返回BEV特征与深度分布估计，上一帧只返回BEV特征
8. 将当前侦与上一帧的BEV特征在通道上进行concat，在BEVDepthHead中进一步提取特征
9. 将不同类别分为多个task处理，利用ceterpointHead完成每个任务的预测结果。（offset_x, offset_y， z， w, l, h, cosA, cosB, vx, vy）
10. 计算3d检测损失：heatmap 类别损失：focal loss， bbox 相关损失：L1 loss
11. 计算深度估计损失：深度估计损失：BCE loss。

### bevdepth中的摄像头内外参编码方式
估计像素深度和摄像头的内参是关联的，所以把摄像头内参编码进网络是非常有意义的。尤其是在multi-view 3d datasets 中
每个camera有不同的FOV, 为了提升深度估计的质量
### 操作流程
- 具体来说，首先使用MLP layer 把camera 的内参维度scaled up到 特征图的维度
- 然后使用Squeeze-and-Excitation module 对image feature 进行re-weight
- 最后concatenate camera extrinsics 和 内参去帮助DepthNet 感知2d 特征在自车坐标系下的空间位置
- $\xi$ denotes the Flatten operation,

- 由于深度估计依赖于相机参数，因此利用mlp将相机参数维度由[6, 27]—>[6, 512, 1, 1]，作为SE注意力权重。
- 使其分别与图像特征进行注意力操作，并利用1*1卷积，进行通道缩放，获得context特征与depth特征
- 对depth特征进行softmax操作获得深度估计分布
- 和LSS思路一样，Context Feature和Depth Distribution作外积获得图像语义特征和深度特征的综合信息。
- GetGeometry()构建视锥点云，并利用相机参数将视锥点云转换到车辆坐标系下的空间位置

利用深度估计网络DepthNet提取环视图像特征的context和depth信息  
操作步骤：
①. 利用mlp将相机参数维度由[6, 27]—>[6, 512, 1, 1]，作为context和depth的注意力系数
②. 分别对图像context和depth进行se注意力操作
③. 利用1*1卷积改变context与depth注意力操作的通道数。
④. 在通道维度进行contact

<img width="302" alt="image" src="https://github.com/Tony-Hou/Learning-AI/assets/26059901/ee79db0d-9701-4abc-beec-a94ffc58ac89">
<img width="555" alt="image" src="https://github.com/Tony-Hou/Learning-AI/assets/26059901/d562a35e-d25f-4ccd-ab2c-a7ccdd2d1bea">

```python
class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x

class SELayer(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)

class DepthNet(nn.Module):

    def __init__(self, in_channels, mid_channels, context_channels,
                 depth_channels):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(mid_channels,
                                      context_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)
        self.bn = nn.BatchNorm1d(27)
        self.depth_mlp = Mlp(27, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.context_mlp = Mlp(27, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            ASPP(mid_channels, mid_channels),
            build_conv_layer(cfg=dict(
                type='DCN',
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                groups=4,
                im2col_step=128,
            )),
            nn.Conv2d(mid_channels,
                      depth_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )

    def forward(self, x, mats_dict):
        intrins = mats_dict['intrin_mats'][:, 0:1, ..., :3, :3]
        batch_size = intrins.shape[0]
        num_cams = intrins.shape[2]
        ida = mats_dict['ida_mats'][:, 0:1, ...]  # 代表图像数据增强
        sensor2ego = mats_dict['sensor2ego_mats'][:, 0:1, ..., :3, :]
        bda = mats_dict['bda_mat'].view(batch_size, 1, 1, 4,
                                        4).repeat(1, 1, num_cams, 1, 1)  # 代表bev 数据增强
        # camera 内参和外参进行concate  sensor2ego是外参
        mlp_input = torch.cat(
            [
                torch.stack(
                    [
                        intrins[:, 0:1, ..., 0, 0],
                        intrins[:, 0:1, ..., 1, 1],
                        intrins[:, 0:1, ..., 0, 2],
                        intrins[:, 0:1, ..., 1, 2],
                        ida[:, 0:1, ..., 0, 0],
                        ida[:, 0:1, ..., 0, 1],
                        ida[:, 0:1, ..., 0, 3],
                        ida[:, 0:1, ..., 1, 0],
                        ida[:, 0:1, ..., 1, 1],
                        ida[:, 0:1, ..., 1, 3],
                        bda[:, 0:1, ..., 0, 0],
                        bda[:, 0:1, ..., 0, 1],
                        bda[:, 0:1, ..., 1, 0],
                        bda[:, 0:1, ..., 1, 1],
                        bda[:, 0:1, ..., 2, 2],
                    ],
                    dim=-1,
                ),
                sensor2ego.view(batch_size, 1, num_cams, -1),
            ],
            -1,
        )
        # 当前帧的相机参数
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        # x 为image feature
        x = self.reduce_conv(x)
        # 利用mlp将相机参数维度由[6, 27]--->[6, 512, 1, 1]，作为context的注意力系数
        context_se = self.context_mlp(mlp_input)[..., None, None]  # MLP layer [6, 512, 1, 1]
        # 注意力操作
        context = self.context_se(x, context_se)                   # SE module
        # 1*1 卷积降维通道数 [6, 512, 16, 44]--->[6, 80, 16, 44]
        context = self.context_conv(context)
        # 利用mlp将相机参数维度由[6, 27]--->[6, 512, 1, 1]，作为depth的注意力系数
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        # 注意力操作
        depth = self.depth_se(x, depth_se)
        # 1*1卷积降维通道数，[6, 512, 16, 44]--->[6, 112, 16, 44]
        depth = self.depth_conv(depth)
        return torch.cat([depth, context], dim=1)
```

```
# Context feature 和depth distribution 作外积
# img_feat_with_depth: [6, 80, 112, 16, 44]
img_feat_with_depth = depth.unsqueeze(1) * depth_feature[:, self.depth_channels:(self.depth_channels + self.output_channels)].unsqueeze(2)

# 相机坐标系转换为车身坐标系，同时将ego下的空间坐标换算到体素坐标
# 相机坐标系转换到车身坐标系
  geom_xyz = self.get_geometry(
      mats_dict['sensor2ego_mats'][:, sweep_index, ...],
      mats_dict['intrin_mats'][:, sweep_index, ...],
      mats_dict['ida_mats'][:, sweep_index, ...],
      mats_dict.get('bda_mat', None),
  )
  # [1, 6, 80, 112, 16, 44]--->[1, 6, 112, 16, 44, 80]
  img_feat_with_depth = img_feat_with_depth.permute(0, 1, 3, 4, 5, 2)
  # ego下的空间坐标转换到体素坐标（范围从-50m~50m，转换为0-200的体素坐标）geom_xyz: [1, 6, 112, 16, 44, 3]
  geom_xyz = ((geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)) /self.voxel_size).int()

```

Efficient Voxel Pooling将特征img_feat_with_depth 放到BEV空间，[b, 80, 128, 128]
为视锥的每个特征点分配一个CUDA thread，它的作用是把特征值加到相应的BEV网格中，利用了GPU计算的并行化完成稠密特征点的池化过程。
```
# geom_xyz：[1, 6, 112, 16, 44, 80]
feature_map = voxel_pooling(geom_xyz, img_feat_with_depth.contiguous(), self.voxel_num.cuda())

void voxel_pooling_forward_kernel_launcher(...){
     dim3 blocks(DIVUP(batch_size * num_points, THREADS_PER_BLOCK)); // 473088 / 128 = 3696 个 block ，排布为 3696*1
     dim3 threads(THREADS_BLOCK_X, THREADS_BLOCK_Y);  // 每个 block中 有 128 个 thread，排布为 32 * 4
     voxel_pooling_forward_kernel<<<blocks, threads, 0, stream>>>(
       batch_size, num_points, num_channels, num_voxel_x, num_voxel_y,
       num_voxel_z, geom_xyz, input_features, output_features, pos_memo);
 }
 ​
 __global__ void voxel_pooling_forward_kernel(...) {
   /*
   Args:
     batch_size:当前block在哪个batch ，假定batchsize==1
     num_points:视锥点个数，473088
     num_channels:特征维度，80
     num_voxel_x:bev特征x大小
     num_voxel_y:bev特征y大小
     geom_xyz:视锥坐标的指针，[1, 473088, 3]
     input_features:输入特征图的指针，[1, 473088, 80]
     output_features:输出特征图的指针，[1, 128, 128, 80]
     pos_memo:记录x,y坐标，[1, 473088, 3]
   */
   # 所有thread 同时计算
   const int bidx = blockIdx.x;   // bidx，当前block在当前grid中x维度的索引
   const int tidx = threadIdx.x;  // tidx，当前thread在当前block中x维度的索引
   const int tidy = threadIdx.y;  // tidy，当前thread在当前block中y维度的索引
   const int sample_dim = THREADS_PER_BLOCK; // sample_dim 128 ,每个block中的thread数量 
   const int idx_in_block = tidy * THREADS_BLOCK_X + tidx;   // 当前thread在当前block中的全局索引
 ​
   const int block_sample_idx = bidx * sample_dim; //当前block在当前grid中的全局索引
   const int thread_sample_idx = block_sample_idx + idx_in_block; // 当前thread在当前grid中的全局索引
     
   const int total_samples = batch_size * num_points; // 总thread数量
 ​
   __shared__ int geom_xyz_shared[THREADS_PER_BLOCK * 3]; // 128 * 3 共享内存，记录一个block中所有点的坐标
 ​
   if (thread_sample_idx < total_samples) {
     // 将一个block中的所有视锥点的坐储存在共享内存geom_xyz_shared中，(所有block同时进行)
     const int sample_x = geom_xyz[thread_sample_idx * 3 + 0];
     const int sample_y = geom_xyz[thread_sample_idx * 3 + 1];
     const int sample_z = geom_xyz[thread_sample_idx * 3 + 2];
     geom_xyz_shared[idx_in_block * 3 + 0] = sample_x;
     geom_xyz_shared[idx_in_block * 3 + 1] = sample_y;
     geom_xyz_shared[idx_in_block * 3 + 2] = sample_z;
     if ((sample_x >= 0 && sample_x < num_voxel_x) &&
         (sample_y >= 0 && sample_y < num_voxel_y) &&
         (sample_z >= 0 && sample_z < num_voxel_z)) {
       pos_memo[thread_sample_idx * 3 + 0] = thread_smple_idx / num_points; // 将z轴变为0
       pos_memo[thread_sample_idx * 3 + 1] = sample_y;  // 保存视锥y坐标
       pos_memo[thread_sample_idx * 3 + 2] = sample_x;  // 保存视锥x坐标
     }
   }
 ​
   __syncthreads();
   // 可以分为两个步骤，1、先找到当前视锥点在output_features,也就是BEV特征下索引，再找到当前视锥点在input_features中的索引，然后再将两个位置的特征进行相加，由于input_features可能出现多个索引对应于output_features中的同一个索引，必须使用原子加 atomicAdd，可以参考上方的示意图
   for (int i = tidy;
        i < THREADS_PER_BLOCK && block_sample_idx + i < total_samples;
        i += THREADS_BLOCK_Y) {
     const int sample_x = geom_xyz_shared[i * 3 + 0];
     const int sample_y = geom_xyz_shared[i * 3 + 1];
     const int sample_z = geom_xyz_shared[i * 3 + 2];
     if (sample_x < 0 || sample_x >= num_voxel_x || sample_y < 0 ||
         sample_y >= num_voxel_y || sample_z < 0 || sample_z >= num_voxel_z) {
       continue;
     }
     const int batch_idx = (block_sample_idx + i) / num_points;
     for (int j = tidx; j < num_channels; j += THREADS_BLOCK_X) {
       atomicAdd(&output_features[(batch_idx * num_voxel_y * num_voxel_x +sample_y * num_voxel_x + sample_x) *num_channels +j],input_features[(block_sample_idx + i) * num_channels + j]);
     }
   }
 }

```
当前帧返回BEV特征与深度分布估计，上一帧只返回BEV特征

将当前侦与上一帧的BEV特征在通道上进行concat，在BEVDepthHead中进一步提取特征
进行3次下采样，并利用second_fpn进行特征融合
```python
# FPN
trunk_outs = [x]
if self.trunk.deep_stem:
    x = self.trunk.stem(x)
else:
    x = self.trunk.conv1(x)
    x = self.trunk.norm1(x)
    x = self.trunk.relu(x)
# 将x[b, 160, 128, 128]三次下采样--->[b, 160， 64， 64]--->[b, 320, 32, 32]--->[b, 640, 16, 16]
for i, layer_name in enumerate(self.trunk.res_layers):
    res_layer = getattr(self.trunk, layer_name)
    x = res_layer(x)
    if i in self.trunk.out_indices:
        trunk_outs.append(x)
# 利用second_fpn将四个不同尺度的特征层融合, fpn_output:[b, 256, 128, 128]
fpn_output = self.neck(trunk_outs)

```

target生成
```python
def get_targets_single(self, gt_bboxes_3d, gt_labels_3d):
    """Generate training targets for a single sample.
    生成单张图片的训练gt
    """
    max_objs = self.train_cfg['max_objs'] * self.train_cfg['dense_reg']
    grid_size = torch.tensor(self.train_cfg['grid_size'])
    pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
    voxel_size = torch.tensor(self.train_cfg['voxel_size'])

    feature_map_size = grid_size[:2] // self.train_cfg['out_size_factor']

    # reorganize the gt_dict by tasks
    task_masks = []
    flag = 0
    for class_name in self.class_names:
        task_masks.append([torch.where(gt_labels_3d == class_name.index(i) + flag) for i in class_name])
        flag += len(class_name)

    task_boxes = []
    task_classes = []
    flag2 = 0
    for idx, mask in enumerate(task_masks):
        task_box = []
        task_class = []
        for m in mask:
            task_box.append(gt_bboxes_3d[m])
            # 0 is background for each task, so we need to add 1 here.
            task_class.append(gt_labels_3d[m] + 1 - flag2)
        task_boxes.append(torch.cat(task_box, axis=0).to(gt_bboxes_3d.device))
        task_classes.append(torch.cat(task_class).long().to(gt_bboxes_3d.device))
        flag2 += len(mask)
    draw_gaussian = draw_heatmap_gaussian
    heatmaps, anno_boxes, inds, masks = [], [], [], []
    # 生成每个task_head需要处理的GT
    for idx, task_head in enumerate(self.task_heads):
        heatmap = gt_bboxes_3d.new_zeros((len(self.class_names[idx]), feature_map_size[1], feature_map_size[0]), device='cuda')
        # anno_box: [500, 10]
        anno_box = gt_bboxes_3d.new_zeros((max_objs, len(self.train_cfg['code_weights'])), dtype=torch.float32, device='cuda')

        ind = gt_labels_3d.new_zeros((max_objs), dtype=torch.int64, device='cuda')
        mask = gt_bboxes_3d.new_zeros((max_objs), dtype=torch.uint8, device='cuda')
        # gt数量：min（gt_nums, 500）
        num_objs = min(task_boxes[idx].shape[0], max_objs)
        # 遍历每个GT，生成对应的GT标签
        for k in range(num_objs):
            cls_id = task_classes[idx][k] - 1

            width = task_boxes[idx][k][3]
            length = task_boxes[idx][k][4]
            width = width / voxel_size[0] / self.train_cfg['out_size_factor']
            length = length / voxel_size[1] / self.train_cfg['out_size_factor']

            if width > 0 and length > 0:
                # 求解高斯半径
                radius = gaussian_radius((length, width), min_overlap=self.train_cfg['gaussian_overlap'])
                radius = max(self.train_cfg['min_radius'], int(radius))

                # x， y， z中心点
                x, y, z = task_boxes[idx][k][0], task_boxes[idx][k][1], task_boxes[idx][k][2]
                # gt 中心点在特征图上坐标
                coor_x = (x - pc_range[0]) / voxel_size[0] / self.train_cfg['out_size_factor']
                coor_y = (y - pc_range[1]) / voxel_size[1] / self.train_cfg['out_size_factor']
                center = torch.tensor([coor_x, coor_y], dtype=torch.float32, device='cuda')
                center_int = center.to(torch.int32)

                # 过滤掉超过特征图尺寸外的点
                if not (0 <= center_int[0] < feature_map_size[0] and 0 <= center_int[1] < feature_map_size[1]):
                    continue

                draw_gaussian(heatmap[cls_id], center_int, radius)

                new_idx = k
                x, y = center_int[0], center_int[1]

                assert y * feature_map_size[0] + x < feature_map_size[0] * feature_map_size[1]
                # GT boxes位置索引
                ind[new_idx] = y * feature_map_size[0] + x
                # 有效GT索引
                mask[new_idx] = 1

                # TODO: support other outdoor dataset
                if len(task_boxes[idx][k]) > 7:
                    vx, vy = task_boxes[idx][k][7:]

                # gtbox旋转角度
                rot = task_boxes[idx][k][6]
                # gtbox的尺寸
                box_dim = task_boxes[idx][k][3:6]
                # 对box尺寸进行log编码
                if self.norm_bbox:
                    box_dim = box_dim.log()

                # 添加vx，vy速度标签，生成anno_box标签
                if len(task_boxes[idx][k]) > 7:
                    anno_box[new_idx] = torch.cat([
                        center - torch.tensor([x, y], device='cuda'),          # 中心点偏移量(offset_x, offset_y)
                        z.unsqueeze(0),                                        # 中心点 z
                        box_dim,                                               # box在特征图上log编码后的尺寸（w，l，h）
                        torch.sin(rot).unsqueeze(0),                           # box旋转角度正弦
                        torch.cos(rot).unsqueeze(0),                           # box旋转角度余弦
                        vx.unsqueeze(0),                                       # x方向速度
                        vy.unsqueeze(0),                                       # y方向速度
                    ])
                else:
                    anno_box[new_idx] = torch.cat([
                        center - torch.tensor([x, y], device='cuda'),
                        z.unsqueeze(0),
                        box_dim,
                        torch.sin(rot).unsqueeze(0),
                        torch.cos(rot).unsqueeze(0)
                    ])

        heatmaps.append(heatmap)
        anno_boxes.append(anno_box)
        masks.append(mask)
        inds.append(ind)
    return heatmaps, anno_boxes, inds, masks

```
利用ceterpoint head完成每个任务的预测结果
计算3d检测损失
heatmap 类别损失：focal loss
bbox 相关损失：L1 loss
```python
def loss(self, targets, preds_dicts, **kwargs):
    heatmaps, anno_boxes, inds, masks = targets
    return_loss = 0
    for task_id, preds_dict in enumerate(preds_dicts):
        # heatmap focal loss
        preds_dict[0]['heatmap'] = clip_sigmoid(preds_dict[0]['heatmap'])
        # 正样本数量
        num_pos = heatmaps[task_id].eq(1).float().sum().item()

        cls_avg_factor = torch.clamp(reduce_mean(heatmaps[task_id].new_tensor(num_pos)), min=1).item()
        # 类别损失: focal loss
        loss_heatmap = self.loss_cls(preds_dict[0]['heatmap'], heatmaps[task_id], avg_factor=cls_avg_factor)
        # GT box: [b, 500, 10]
        target_box = anno_boxes[task_id]
        # 将任务头bbox框的预测结果进行concat，与GT对齐， [offset_x， offset_y，z, w，l，h, sin(rot), cos(rot), vel_x, vel_y]
        if 'vel' in preds_dict[0].keys():
            preds_dict[0]['anno_box'] = torch.cat(
                (preds_dict[0]['reg'], preds_dict[0]['height'],
                 preds_dict[0]['dim'], preds_dict[0]['rot'],
                 preds_dict[0]['vel']),
                dim=1,
            )
        else:
            preds_dict[0]['anno_box'] = torch.cat(
                (preds_dict[0]['reg'], preds_dict[0]['height'],
                 preds_dict[0]['dim'], preds_dict[0]['rot']),
                dim=1,
            )
        # gt数量
        num = masks[task_id].float().sum()

        ind = inds[task_id]
        # preds_dict[0]['anno_box']: [b, 10, 128, 128]--->[b, 128, 128, 10]
        pred = preds_dict[0]['anno_box'].permute(0, 2, 3, 1).contiguous()
        # pred: [b, 128, 128, 10]--->[b, 16384, 10]
        pred = pred.view(pred.size(0), -1, pred.size(3))
        # 只保留ind索引对应的max_obj个预测结果。  pred: [b, 16384, 10]--->[b, 500, 10]
        pred = self._gather_feat(pred, ind)
        # 将mask维度扩展成和target_box一致
        mask = masks[task_id].unsqueeze(2).expand_as(target_box).float()
        # gtbox 数量
        num = torch.clamp(reduce_mean(target_box.new_tensor(num)), min=1e-4).item()
        # 过滤nan结果
        isnotnan = (~torch.isnan(target_box)).float()
        mask *= isnotnan
        # 对应[offset_x， offset_y，z, w，l，h, sin(rot), cos(rot), vel_x, vel_y] 十个权重
        code_weights = self.train_cfg['code_weights']
        # bbox不同属性的权重
        bbox_weights = mask * mask.new_tensor(code_weights)
        # L1 loss求解bbox 损失
        loss_bbox = self.loss_bbox(pred, target_box, bbox_weights, avg_factor=num)

        return_loss += loss_bbox
        return_loss += loss_heatmap
    return return_loss

```
计算深度估计损失
深度估计损失：BCE loss
```python
def get_depth_loss(self, depth_labels, depth_preds):
    # 对depth_labels 1/16下采样换算到特征图，[1, 6, 256, 704]--->[1*6*16*44, 112]--->[4224, 112]
    depth_labels = self.get_downsampled_gt_depth(depth_labels)
    # depth_preds: [6, 112, 16, 44]--->[6, 16, 44, 112]--->[4224, 112]
    depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(-1, self.depth_channels)
    fg_mask = torch.max(depth_labels, dim=1).values > 0.0
    # 计算深度估计损失： BCE loss
    with autocast(enabled=False):
        depth_loss = (F.binary_cross_entropy(
            depth_preds[fg_mask],
            depth_labels[fg_mask],
            reduction='none',
        ).sum() / max(1.0, fg_mask.sum()))

    return 3.0 * depth_loss

```

# 传入的参数shape:
# sensor2ego_mat(1, 6, 4, 4), intrin_mat(1, 6, 4, 4), ida_mat(1, 6, 4, 4), bda_mat(1, 4, 4)

[bevdepth](https://blog.csdn.net/weixin_42454048/article/details/130516392)
