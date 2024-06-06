# Nuscenes datasets

```python
Total Scene  Num: 850
Total Sample Num: 34149
[ expo_time_bet_adj_cams] Avg:    8.52ms  2STD:    8.53ms
[max_delta_time_bet_cams] Avg:   42.61ms  2STD:   42.67ms
[       cam_refresh_time] Avg:  498.93ms  2STD:  503.95ms
[     lidar_refresh_time] Avg:  498.93ms  2STD:  503.97ms
[   delta_lidar_cam_time] Avg:    0.99ms  2STD:    2.37ms

```
![image](https://github.com/Tony-Hou/Learning-AI/assets/26059901/d5ca1509-0d33-4a34-ba56-30467d753681)

### 3D box 
nuscenes 的3d box是在全局坐标系下的，3d rotation是用四元数（w, x, y, z）表示

表示rotation的四元数可以利用Python包pyquaternion转换成(pitch，yaw，roll)的形式，而且可以计算对应的旋转矩阵和逆矩阵

由于每张图像的时间戳、激光的时间戳都两两不相同，它们有各自的位姿补偿（ego data），进行坐标系转换的时候需要注意一下
得到激光雷达的“ego_pose”的参数，该ego_pose就是将ego坐标转到global坐标的。因此，求其逆，将global坐标转到ego坐标；然后，通过lidar的calibrated矩阵（将传感器坐标转到ego坐标），求其逆，将ego坐标转到lidar传感器坐标；



```python
# 标注真值到激光坐标系
ann = nusc.get('sample_annotation', token)
calib_data = nusc.get('calibrated_sensor', lidar_data['calibrated_sensor_token'])
ego_data = nusc.get('ego_pose', lidar_data['ego_pose_token'])
# global frame
center = np.array(ann['translation'])
orientation = np.array(ann['rotation'])
# 从global frame转换到ego vehicle frame
quaternion = Quaternion(ego_data['rotation']).inverse
center -= np.array(ego_data['translation'])
center = np.dot(quaternion.rotation_matrix, center)
orientation = quaternion * orientation
# 从ego vehicle frame转换到sensor frame
quaternion = Quaternion(calib_data['rotation']).inverse
center -= np.array(calib_data['translation'])
center = np.dot(quaternion.rotation_matrix, center)
orientation = quaternion * orientation

```
标注真值(global frame)投影到图像(pixel coord)：使用位姿补偿转换到车身坐标系，然后再根据相机外参转换到相机坐标系，
最后使用相机内参得到像素坐标系下的坐标。标注真值到车身坐标系的过程和上面类似，不过calib_data和ego_data需要从camera_data中获取，
得到标注3D框在相机坐标系下的角点坐标points后，然后再使用相机内参投影至图像。



激光真值(lidar frame)投影至图像(pixel coord)就相对麻烦一点，因为图像和激光时间戳不一致，需要多进行一步时间戳的变换
```python
# step1: lidar frame -> ego frame
calib_data = nusc.get('calibrated_sensor', lidar_file['calibrated_sensor_token'])
rot_matrix = Quaternion(calib_data['rotation']).rotation_matrix
points[:3, :] = np.dot(rot_matrix, points[:3, :])
for i in range(3):
    points[i, :] += calib_data['translation'][i]

# step2: ego frame -> global frame
ego_data = nusc.get('ego_pose', lidar_file['ego_pose_token'])
rot_matrix = Quaternion(ego_data['rotation']).rotation_matrix
points[:3, :] = np.dot(rot_matrix, points[:3, :])
for i in range(3):
    points[i, :] += ego_data['translation'][i]

# step3: global frame -> ego frame
ego_data = nusc.get('ego_pose', camera_data['ego_pose_token'])
for i in range(3):
    points[i, :] -= ego_data['translation'][i]
rot_matrix = Quaternion(ego_data['rotation']).rotation_matrix.T
points[:3, :] = np.dot(rot_matrix, points[:3, :])

# step4: ego frame -> cam frame
calib_data = nusc.get('calibrated_sensor', camera_data['calibrated_sensor_token'])
for i in range(3):
    points[i, :] -= calib_data['translation'][i]
rot_matrix = Quaternion(calib_data['rotation']).rotation_matrix.T
points[:3, :] = np.dot(rot_matrix, points[:3, :])

# step5: cam frame -> uv pixel
visible = points[2, :] > 0.1
colors = get_rgb_by_distance(points[2, :], min_val=0, max_val=50)
intrinsic = calib_data['camera_intrinsic']
trans_mat = np.eye(4)
trans_mat[:3, :3] = np.array(intrinsic)
points = np.concatenate((points[:3, :], np.ones((1, points.shape[1]))), axis=0)
points = np.dot(trans_mat, points)[:3, :]
points /= points[2, :]
points = points[:2, :]

```

# Nuscenes 数据集3D box 坐标系定义

https://www.nuscenes.org/tutorials/prediction_tutorial.html

# KITTI 数据集3D box坐标系




