---
tags: [slam, localization, mapping]
created: "2026-07-26"
---

# 🗺️ SLAM 与定位

> SLAM = Simultaneous Localization And Mapping（同时定位与建图）。
> 小车"知道自己在哪"的基础能力。

---

## 一、问题定义

小车需要回答三个问题：

| 问题 | 英文 | 依赖 |
|------|------|------|
| 我在哪？ | Localization | 里程计 + IMU + 视觉/LiDAR |
| 周围有什么？ | Perception | 摄像头 / 超声波 / LiDAR |
| 我该怎么走？ | Planning | 定位 + 地图 + 感知 |

SLAM 解决了"建图"和"定位"的鸡生蛋问题：
- 要定位需要地图，但地图需要定位才能建
- SLAM 同时做两者：边建图边定位

---

## 二、里程计 (Odometry)

### 编码器里程计（最基础）

从左右轮编码器推算位姿：

$$\begin{aligned}
\Delta s &= \frac{\Delta s_L + \Delta s_R}{2} \\
\Delta \theta &= \frac{\Delta s_R - \Delta s_L}{W} \\
x_{t+1} &= x_t + \Delta s \cdot \cos(\theta_t + \frac{\Delta \theta}{2}) \\
y_{t+1} &= y_t + \Delta s \cdot \sin(\theta_t + \frac{\Delta \theta}{2}) \\
\theta_{t+1} &= \theta_t + \Delta \theta
\end{aligned}$$

其中 $W$ 是左右轮间距（轮距），$\Delta s_L, \Delta s_R$ 是左右轮位移。

```python
import math

class DifferentialDriveOdom:
    def __init__(self, wheel_base_mm):
        self.wheel_base = wheel_base_mm / 1000.0  # 米
        self.x, self.y, self.theta = 0.0, 0.0, 0.0
    
    def update(self, delta_left_m, delta_right_m):
        """输入左右轮位移（米），更新位姿"""
        ds = (delta_left_m + delta_right_m) / 2.0
        dtheta = (delta_right_m - delta_left_m) / self.wheel_base
        
        self.x += ds * math.cos(self.theta + dtheta / 2)
        self.y += ds * math.sin(self.theta + dtheta / 2)
        self.theta += dtheta
        self.theta = math.atan2(math.sin(self.theta), math.cos(self.theta))  # 归一化
```

### 编码器误差累积

编码器里程计的致命问题：误差无限累积。

| 误差来源 | 影响 |
|----------|------|
| 轮径不准 | 距离 scale 误差 |
| 轮距不准 | 转角 scale 误差 |
| 轮胎打滑 | 位移偏小 |
| 路面不平 | 随机误差 |
| 编码器分辨率 | 量化噪声 |

> 纯编码器跑 10 米，误差可达 30cm-1m。必须融合 IMU / 视觉 / LiDAR 校正。

---

## 三、IMU + 编码器融合（EKF）

### 扩展卡尔曼滤波 (EKF)

EKF 是机器人定位最常用的传感器融合算法：

```
预测步骤（高频，靠 IMU + 编码器）：
  x̂(k|k-1) = f(x̂(k-1), u(k))     # 运动模型预测新位姿
  P(k|k-1) = F·P(k-1)·F^T + Q    # 协方差传播

更新步骤（低频，靠外部观测）：
  K = P·H^T·(H·P·H^T + R)^(-1)  # 卡尔曼增益
  x̂(k) = x̂(k|k-1) + K·(z - h(x̂)) # 状态更新
  P(k) = (I - K·H)·P(k|k-1)      # 协方差更新
```

### 小车上的 EKF 实现思路

```python
# 状态向量: [x, y, θ, v, ω]
# 控制输入: [v_cmd, ω_cmd]  (来自编码器/IMU)
# 观测: [x_gps(可选), y_gps, θ_mag(磁力计航向)]

class EKFLocalization:
    def __init__(self):
        self.x = np.zeros(5)   # 状态
        self.P = np.eye(5) * 0.1  # 协方差
        self.Q = np.diag([0.01, 0.01, 0.01, 0.1, 0.1])  # 过程噪声
        self.R = np.diag([0.5, 0.5, 0.1])                # 观测噪声
    
    def predict(self, v, omega, dt):
        """运动模型预测"""
        x, y, theta, _, _ = self.x
        self.x[0] += v * math.cos(theta) * dt
        self.x[1] += v * math.sin(theta) * dt
        self.x[2] += omega * dt
        # 更新 P... (略，实际用 filterpy 库)
    
    def update_imu_heading(self, heading):
        """用 IMU 磁力计修正航向"""
        z = heading
        h = self.x[2]
        y = z - h  # 创新
        # K = P·H^T / (H·P·H^T + R)
        # x = x + K·y
        # P = (I-KH)·P
    
    def update_visual(self, x_obs, y_obs):
        """用视觉特征/回环修正位置"""
        # 同上
```

### 推荐库：用 `robot_localization`（ROS2 包）

手写 EKF 容易出错，ROS2 的 `robot_localization` 是开箱即用的方案：

```yaml
# ekf.yaml — robot_localization 配置
ekf_filter_node:
  ros__parameters:
    frequency: 30.0
    sensor_timeout: 0.1
    two_d_mode: true          # 2D 平面小车
    
    # 输入话题
    odom0: /encoder/odom      # 编码器里程计
    imu0: /imu/data           # IMU
    odom0_config: [true, true, false, false, false, true, ...]  # 哪些轴用编码器
    imu0_config: [false, false, false, true, true, true, ...]   # IMU 提供哪些观测
```

```bash
# 安装
sudo apt install ros-humble-robot-localization

# 启动
ros2 launch robot_localization ekf.launch.py
```

---

## 四、SLAM 方案选型

### 2D LiDAR SLAM（最稳，推荐）

| 方案 | 特点 | 适用场景 |
|------|------|----------|
| **slam_toolbox** | ROS2 原生，在线+离线 | L3+ 室内/结构化环境 |
| **Cartographer** | Google 的，闭环优秀 | 大型室内环境 |
| **Hector SLAM** | 不需要里程计 | 仅 LiDAR，但需要高更新率 |

```bash
# slam_toolbox 在线建图
sudo apt install ros-humble-slam-toolbox
ros2 launch slam_toolbox online_async_launch.py

# 保存地图
ros2 run nav2_map_server map_saver_cli -f ~/my_map

# 之后用已有地图定位（AMCL）
ros2 launch nav2_bringup localization_launch.py map:=~/my_map.yaml
```

### 纯视觉 SLAM（只用摄像头）

| 方案 | 特点 |
|------|------|
| **ORB-SLAM3** | 学术标杆，支持单目/双目/RGB-D |
| **OpenVSLAM** | ORB-SLAM 的现代化实现 |
| **VINS-Mono** | 视觉+IMU 紧耦合 |

```bash
# ORB-SLAM3 对小车算力要求高（需要 Jetson 级别）
# 对 L2 小车不推荐纯视觉 SLAM——特征点少（地面纹理弱），漂移严重
```

### 小车 SLAM 推荐路径

| 阶段 | 方案 | 理由 |
|:--:|------|------|
| L1-L2 | 不需要 SLAM | 循迹/车道保持用不到全局定位 |
| L3 | 编码器 + IMU (EKF) | 知道大致位置即可，不建图 |
| L3+ | 加 2D LiDAR → slam_toolbox | 建图后做路径规划 |
| L4 | 2D LiDAR + 回环检测 | 实现"从 A 到 B 自主导航" |

---

## 五、回环检测

### 为什么需要回环？

编码器/IMU 误差累积导致轨迹漂移。回环检测（Loop Closure）在识别到"回到了来过的地方"时，强制修正全局位姿。

```
漂移前的地图:          回环修正后:
┌────┐                 ┌────┐
│    │                 │  ┌─┘
│    └──┐              │  │
│       │    →         │  │
│    ┌──┘              │  └─┐
│    │                 │    │
└────┘                 └────┘
```

### 小车的回环方法

| 方法 | 难度 | 效果 |
|------|:--:|------|
| 视觉词袋（ORB-SLAM 内置） | 中 | 需要丰富纹理 |
| LiDAR Scan Context | 中 | 2D/3D 都可 |
| WiFi / BLE 指纹 | 低 | 精度粗但鲁棒 |
| AprilTag 地标 | 低 | 需要贴标签在地面 |
| **手动标记** | 最低 | 按键标记当前位置，再次经过时校正 |

---

## 🔗 关联笔记

- [[传感器集成]] — 编码器、IMU 驱动
- [[路径规划与控制]] — 定位到位姿后，规划路径
- [[3D视觉与投影几何]] — 视觉 SLAM 的几何基础
- [[数据闭环总览]] — SLAM 数据是数据飞轮的输入之一
