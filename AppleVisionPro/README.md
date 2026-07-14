# BRX042501 Apple Vision Pro 遥操作与 LeRobot 数据采集

本目录是 BRX042501 的独立集成层，目标数据链路为：

```text
Apple Vision Pro / visionOS hand tracking
        │  双手 position + quaternion + pinch
        ▼
AppleVisionPro/teleop_bridge.py           HTTP :8899
        │  20-D 双手末端绝对位姿目标
        ▼
AppleVisionPro/brx_control_server.py      HTTP :8765
        │  小型 DLS 位姿适配器 → joint-position target
        ▼
Isaac Lab + BRX042501 (固定 23-D 对外 ABI)
        │
        ├── 4 路同步 RGB
        ├── observation.state = 实际 qpos[23]
        └── action = 实际下发的 joint target[23]
                ▼
        LeRobot v2.1 或 v3 数据集
```

实现遵守以下边界：没有修改 Isaac Lab core、官方 simulator logic 或现有训练流水线；主要实现全部位于 `AppleVisionPro/`。为了保留原启动命令，只把 `scripts/custom/brx_control_server.py` 改成了很薄的兼容入口。原始 `BRX042501_wheel.urdf` 没有改动，四相机版本是独立的 `BRX042501_wheel_4cams.urdf`。

## 1. 已实现内容与当前边界

已经实现：

- Vision Pro 双手位置、姿态和 pinch 输入；四元数输入约定为 ARKit 常用的 `[x, y, z, w]`。
- 首包自动标定或显式重新标定；`--scale 1.0` 时手部位移按米进行 1:1 映射。
- 左右手、左右夹爪、位移/旋转步长、工作空间、跟踪超时和 clutch 安全保持。
- 控制服务对外保持严格的 23 维顺序，最终执行方式是 joint position target。
- 严格四路同步 RGB：头部左、头部右、左腕、右腕。
- 图像、实际关节位置和实际关节命令在同一仿真采样点入队。
- LeRobot 官方 `create → add_frame → save_episode → finalize/consolidate` 路径；支持安装了 v2.1 或 v3 的环境。
- 物理、viewport/WebRTC render 和数据相机使用独立频率；默认墙钟实时节流，不再无限高速 render。
- 运行状态、录制队列丢帧数和实时因子可通过 HTTP 查询。

本仓库没有 visionOS/Swift 客户端工程。已有或新建的 Vision Pro 客户端只要按本文的 JSON 协议向 Python bridge 发包即可。当前主路径是无额外中间件依赖的 HTTP Python bridge；没有猜测一个未知的 ROS2 topic/message ABI。如果后续确定 ROS2 消息定义，可在 bridge 输入侧增加 subscriber，控制服务、相机和数据集部分不需要改变。

控制链中只保留了一个逐步 DLS 末端位姿到关节目标适配器；没有 MPC、阻抗控制或轨迹优化。它用于快速打通 `pose → joint target`，最终下发与记录的仍是绝对 joint position command。

## 2. 文件结构

```text
AppleVisionPro/
├── __init__.py
├── joint_contract.py       # 唯一的 23-D 关节顺序与四相机语义契约
├── camera_manager.py       # 严格四相机、同步帧缓存、按需 PNG
├── brx_control_server.py   # Isaac Lab 仿真、控制、状态、相机、录制 API
├── teleop_bridge.py        # Vision Pro 标定、坐标映射、夹爪、看门狗
├── lerobot_recorder.py     # LeRobot v2/v3 异步 episode recorder
├── validate_dataset.py     # 元数据/文件/官方 loader 数据集验收
├── smoke_test.py           # 不移动机器人的在线四相机和 23-D 验收
├── tests/                  # 不启动 Isaac Sim 的契约单元测试
└── README.md

scripts/custom/brx_control_server.py      # 保留旧命令的兼容启动器
BRX042501/BRX042501_wheel_4cams.urdf      # 四相机 marker 版本，原 URDF 未改
```

不要在别的脚本里复制关节列表。任何训练、回放或策略适配都应从 `AppleVisionPro.joint_contract.JOINT_NAMES_23` 读取或逐项保持完全一致。

## 3. 23 维关节 ABI

`observation.state`、`action`、`GET /state`、`POST /command/joint23` 使用完全相同的顺序：

| index | joint | 语义 |
|---:|---|---|
| 0 | `FoldingModularJoint02_Joint` | 底盘/折叠模块 |
| 1 | `FoldingModularJoint03_Joint` | 底盘/折叠模块 |
| 2 | `Trunk_Joint` | 躯干 |
| 3 | `ArmL02_Joint` | 左臂 1 |
| 4 | `ArmL03_Joint` | 左臂 2 |
| 5 | `ArmL04_Joint` | 左臂 3 |
| 6 | `ArmL05_Joint` | 左臂 4 |
| 7 | `ArmL06_Joint` | 左臂 5 |
| 8 | `ArmL07_Joint` | 左臂 6 |
| 9 | `ArmL08_Joint` | 左臂 7 |
| 10 | `JawBlock01_Joint` | **物理右夹爪** jaw 1 |
| 11 | `JawBlock02_Joint` | **物理右夹爪** jaw 2 |
| 12 | `ArmR02_Joint` | 右臂 1 |
| 13 | `ArmR03_Joint` | 右臂 2 |
| 14 | `ArmR04_Joint` | 右臂 3 |
| 15 | `ArmR05_Joint` | 右臂 4 |
| 16 | `ArmR06_Joint` | 右臂 5 |
| 17 | `ArmR07_Joint` | 右臂 6 |
| 18 | `ArmR08_Joint` | 右臂 7 |
| 19 | `JawBlock03_Joint` | **物理左夹爪** jaw 1 |
| 20 | `JawBlock04_Joint` | **物理左夹爪** jaw 2 |
| 21 | `Head02_Joint` | 头部 1 |
| 22 | `Head03_Joint` | 头部 2 |

特别注意：历史 ABI 中夹爪所在位置看起来是“交叉”的。索引 `10–11` 是物理右夹爪，索引 `19–20` 是物理左夹爪。代码已经按 URDF parent link 验证此映射，不要凭数组相邻位置交换它们。

四个 jaw 的 URDF limit 都是 `[0, 0.041] m`。根据 joint origin 和相反的 axis，`q=0` 是张开，`q=0.041` 是闭合。Vision Pro `pinch=0` 映射到张开，pinch 达到阈值时映射到闭合。

轮子等另外 10 个 non-fixed joints 仍存在于 Isaac Lab articulation，但不属于这个 23-D 学习/控制 ABI；服务会保持它们当前目标，不会把它们混入训练向量。

## 4. 四相机 URDF 分析与运行时设计

解析结果：原 URDF 与四相机派生 URDF 都有 45 个 link、44 个 joint、33 个 non-fixed joint。URDF 中没有可由 Isaac Lab 直接使用的 `<sensor>`/`<camera>` 标签，而是已经具备合理的相机安装 marker links。运行时相机严格附着到这些 links：

| 数据键 | URDF marker link | parent link | URDF joint origin | 用途 |
|---|---|---|---|---|
| `observation.images.head_left` | `EyeL_Link` | `Head03_Link` | `xyz=-0.058536 -0.082985 -0.047450` | 头部左目/全局工作区 |
| `observation.images.head_right` | `EyeR_Link` | `Head03_Link` | `xyz=-0.058536 -0.082985 0.012550` | 头部右目/全局工作区 |
| `observation.images.left_wrist` | `HandCam02_Link` | `LinearclampinggripperJZ02_Link` | `xyz=-0.015451 -0.001231 0.085282` | 左手抓取区域和左夹爪 |
| `observation.images.right_wrist` | `HandCam01_Link` | `LinearclampinggripperJZ01_Link` | `xyz=-0.015451 -0.001231 0.085282` | 右手抓取区域和右夹爪 |

左右眼 joint 的 `rpy` 相同，两个 origin 的距离是 `0.060 m`，处于合理的人眼基线范围。因此没有盲目改动 source pose。`EyeM_Link` 只是机械 marker，明确不生成第五路相机。

Isaac Lab 运行时只创建以下四个 sensor prim：

```text
/World/Cameras/Cam00HeadLeft/CameraSensor
/World/Cameras/Cam01HeadRight/CameraSensor
/World/Cameras/Cam02LeftWrist/CameraSensor
/World/Cameras/Cam03RightWrist/CameraSensor
```

默认参数是 `640×360`、RGB only、focal length `12 mm`、horizontal aperture `24 mm`，水平视场角约 90°。`--camera_pose_mode link` 直接使用 URDF marker 的 `+X forward, +Z up` 姿态，这是最终采集的首选模式。

如果首次服务器渲染发现某个 wrist marker 的实际 mesh 坐标约定不符合预期，可临时使用：

```bash
--camera_pose_mode lookat \
--head_camera_forward 0.25 0 0 \
--wrist_camera_forward 0.20 0 -0.12
```

先用 look-at 确认语义和遮挡，再根据真实画面决定是否调整 URDF；不要仅凭坐标数字修改相机。每次调整后必须运行本文的四相机 smoke test，并人工检查夹爪是否出现在对应 wrist view 的合理边缘位置。

## 5. 服务器启动

以下命令全部相对于 IsaacLab 仓库根目录。服务器上的绝对路径不同不影响使用。

### 5.1 首次无录制启动

建议避开有 Xorg 和其他模型服务的 GPU0，例如用 GPU1：

```bash
cd /home/kemove/zzk_data/IsaacLab

LIVESTREAM=2 ./isaaclab.sh \
  -p scripts/custom/brx_control_server.py \
  --urdf_path BRX042501/BRX042501_wheel_4cams.urdf \
  --force_usd_conversion \
  --no_instanceable \
  --enable_cameras \
  --device cuda:1 \
  --host 127.0.0.1 \
  --sim_dt 0.0333333333 \
  --render_hz 30 \
  --camera_hz 15
```

说明：

- `--device cuda:1` 让 Isaac Lab 的 physics/render 选择 GPU1；先用 `nvidia-smi` 确认该卡空闲。
- 第一次或 URDF 修改后使用 `--force_usd_conversion`。确认 USD 正常后可去掉它，缩短启动时间。
- `--host 127.0.0.1` 是推荐拓扑：bridge 与 Isaac Lab 在同一服务器，只有 bridge 的 8899 端口暴露给 Vision Pro。
- `--sim_dt 1/30` 保持 30 Hz physics；默认开启墙钟 pacing。不要加 `--no_realtime`，除非做离线吞吐测试。
- `--render_hz` 控制 viewport/WebRTC 更新；`--camera_hz` 控制四路训练图像采样，两者不再绑死。
- 启动时会做 4 次有限 render warm-up，不会形成无限 render loop。

服务启动成功后日志应包含：

```text
[BRX] URDF contract OK: ... non_fixed=33, stereo_baseline=0.060m
[BRX][camera] head_left: ...
[BRX][camera] head_right: ...
[BRX][camera] left_wrist: ...
[BRX][camera] right_wrist: ...
[BRX] rates: physics=30.0Hz render=30.0Hz camera=15.0Hz realtime=True
[BRX] Setup complete.
```

每隔约 2 秒还会打印：

```text
[BRX][perf] physics=...Hz render=...Hz camera=...Hz rtf=...
```

正常目标是 physics 接近 30 Hz、camera 接近 15 Hz、`rtf` 接近 1。若计算量超过实时能力，CPU/GPU 仍可能满载，此时应降低 render/camera 频率或分辨率，而不是关闭 pacing。

### 5.2 带 LeRobot 录制启动

LeRobot 必须安装在 `./isaaclab.sh -p` 实际使用的 Python 环境中。先检查：

```bash
./isaaclab.sh -p -c "import lerobot; print(lerobot.__version__)"
```

如未安装，请按照你要使用的 LeRobot v2.1 或 v3 官方环境安装；本集成不强制 pin 一个版本，也不会偷偷引入依赖。然后启动：

```bash
LIVESTREAM=2 ./isaaclab.sh \
  -p scripts/custom/brx_control_server.py \
  --urdf_path BRX042501/BRX042501_wheel_4cams.urdf \
  --no_instanceable \
  --enable_cameras \
  --device cuda:1 \
  --host 127.0.0.1 \
  --sim_dt 0.0333333333 \
  --render_hz 30 \
  --camera_hz 15 \
  --record_root datasets/brx_vp_pick_red_001 \
  --record_repo_id local/brx_vp_pick_red_001 \
  --record_fps 15 \
  --lerobot_format auto
```

约束：

- `record_fps` 必须小于等于有效 `camera_hz`。
- 一个已 `finalize` 的 root 不可继续写；下一次采集使用新的 root，防止覆盖或混合 session。
- 默认相机 feature 是 `video`，由 LeRobot 写入 MP4。加 `--record_images` 可改为 image feature。
- `--lerobot_format auto` 使用已安装 LeRobot 的原生格式。`v2`/`v3` 选项是格式断言，不是格式转换器；不匹配时会拒绝开始。
- 若要同时交付 v2.1 和 v3，推荐分别使用对应 LeRobot 环境录制/迁移，而不是在同一 root 混写。

也可以加 `--record_task "Pick the red block and place it in the bucket."` 在服务器 ready 后自动开始第一个 episode；日常遥操作更建议由 Vision Pro 的 `record` 边沿控制。

## 6. Vision Pro bridge 启动与标定

第一轮机械安全测试建议先只开放位置、缩小比例：

```bash
./isaaclab.sh -p AppleVisionPro/teleop_bridge.py \
  --brx_url http://127.0.0.1:8765 \
  --listen_host 0.0.0.0 \
  --listen_port 8899 \
  --axis_map x,y,z \
  --scale 0.3 \
  --position_only
```

确认左右手、正负方向、夹爪和工作空间都正确后，切换到最终 1:1 位姿模式：

```bash
./isaaclab.sh -p AppleVisionPro/teleop_bridge.py \
  --brx_url http://127.0.0.1:8765 \
  --listen_host 0.0.0.0 \
  --listen_port 8899 \
  --axis_map x,y,z \
  --scale 1.0 \
  --rate_hz 30 \
  --max_step_m 0.04 \
  --max_rotation_step_deg 12
```

Vision Pro 连接：`http://<服务器局域网IP>:8899/teleop`。

标定规则：

1. 操作者双手放在舒适中立位置，机器人也处于安全初始姿态。
2. 第一份双手 tracking 均有效的数据包会自动标定。
3. 标定记录“当前手位姿 ↔ 当前机器人左右 EE 位姿”，后续只映射相对位移/相对旋转，因此不会要求两个世界原点相同。
4. 发送 `"calibrate": true` 可随时重新标定。重新标定时机器人应静止，手不要贴近工作空间边界。

`--axis_map` 定义：三个 token 依次表示机器人 base 的 x/y/z 各取 Vision 坐标哪个轴。例如 `z,x,y` 表示 `robot_delta=[vision_z, vision_x, vision_y]`。为了使姿态变换有效，映射必须是右手系旋转，行列式为 `+1`；反射映射会在启动时直接报错。先用 identity `x,y,z`，观察实际 visionOS app 提供的 anchor 坐标后再调整。

默认安全限制：

- 单包平移最大 `0.04 m`。
- 单包旋转最大 `12°`。
- base-frame z 工作区 `[0.35, 1.35] m`，x/y 默认 `[-1.2, 1.2] m`。
- 两只手都必须 `tracking=true`，位置/四元数必须有限且四元数非零。
- `clutch=false` 或 `enabled=false` 时立即向服务发送 hold。
- `0.35 s` 没有收到双手均有效的新包时，机器人保持当前关节位置；如果该 bridge 启动了录制，当前 episode 会自动保存并停止。
- sequence 不递增的包会丢弃；输入超过 30 Hz 的包会限频。

## 7. Vision Pro JSON 协议

每个包：

```json
{
  "sequence": 1001,
  "calibrate": false,
  "clutch": true,
  "record": true,
  "task": "Pick the red block and place it in the bucket.",
  "left": {
    "tracking": true,
    "position": [0.12, 1.05, -0.35],
    "quaternion": [0.0, 0.0, 0.0, 1.0],
    "pinch": 0.0
  },
  "right": {
    "tracking": true,
    "position": [-0.15, 1.02, -0.33],
    "quaternion": [0.0, 0.0, 0.0, 1.0],
    "pinch": 0.0
  }
}
```

字段说明：

- `position`：米，必须在同一个稳定 ARKit/RealityKit anchor frame 中。
- `quaternion`：`[x,y,z,w]`，bridge 内部归一化并转旋转矩阵。
- `pinch`：归一化闭合强度；0 为张开，达到默认阈值 0.75 即完全闭合。
- `record`：边沿语义。第一次从 false/未设置变 true 时开始 episode；true→false 时保存 episode。
- `task`：每个 episode 必需的自然语言任务。只有 record 上升沿使用它。
- `sequence`：建议单调递增，防止网络乱序。
- `calibrate`：只在需要重置相对参考帧的那一包设 true。

调试时可从普通电脑发送一包，但它会产生运动命令，必须先清空机器人周围：

```bash
curl -sS -X POST http://127.0.0.1:8899/teleop \
  -H 'Content-Type: application/json' \
  -d '{
    "sequence": 1,
    "calibrate": true,
    "record": false,
    "left":{"tracking":true,"position":[0,0,0],"quaternion":[0,0,0,1],"pinch":0},
    "right":{"tracking":true,"position":[0,0,0],"quaternion":[0,0,0,1],"pinch":0}
  }'
```

录制也可独立控制：

```bash
curl -sS -X POST http://127.0.0.1:8899/record/start \
  -H 'Content-Type: application/json' \
  -d '{"task":"Pick the red block and place it in the bucket."}'

curl -sS -X POST http://127.0.0.1:8899/record/stop \
  -H 'Content-Type: application/json' -d '{}'
```

`/record/abort` 会尝试清除当前未保存 episode。只有安装的 LeRobot 版本提供安全的 `clear_episode_buffer` 时才允许 abort，否则会报错，避免意外污染后续 episode。

## 8. 控制服务 API

只读接口：

| method/path | 内容 |
|---|---|
| `GET /health` | ready、physics/render/camera FPS、RTF、录制状态 |
| `GET /state` | `qpos23`、`action23`、joint names、双手 EE、相机/录制状态 |
| `GET /camera` | 同步 frame id、时间戳、可用相机、尺寸和实测 capture FPS |
| `GET /record/status` | episode 帧数、总帧数、队列、drop、格式版本 |
| `GET /camera/head_left.png` | 当前头部左图 |
| `GET /camera/head_right.png` | 当前头部右图 |
| `GET /camera/left_wrist.png` | 当前左腕图 |
| `GET /camera/right_wrist.png` | 当前右腕图 |

`GET /camera/head.png` 只作为旧工具兼容 alias，等同 `head_left`，不构成第五路相机。

运动接口：

- `POST /command/ee6d`：20 维绝对 base-frame 目标：`[left xyz(3), left rot6d(6), left gripper(1), right xyz(3), right rot6d(6), right gripper(1)]`。
- `POST /command/joint23`：`{"qpos":[23]}`，绝对 joint target。
- `POST /command/reset_joint23`：瞬时 reset，仅用于受控测试/回放初始化，不用于日常遥操作。
- `POST /command/stop`：保持当前关节位置。
- `POST /command/gripper`：只更新左右 gripper 标量。

录制接口：`POST /record/start`、`/record/stop`、`/record/abort`、`/record/finalize`。`finalize` 后该 server session 的 dataset 只读，不可开始新 episode。

服务默认没有认证，不应直接暴露到公网。推荐只监听 localhost，由同机 bridge 暴露在受信任局域网，并用系统防火墙只允许 Vision Pro 所在网段访问 8899。

## 9. LeRobot 数据内容与同步语义

每个 frame 至少包含：

```text
observation.images.head_left    uint8 [H,W,3] / video
observation.images.head_right   uint8 [H,W,3] / video
observation.images.left_wrist   uint8 [H,W,3] / video
observation.images.right_wrist  uint8 [H,W,3] / video
observation.state               float32 [23]
action                          float32 [23]
task                            string
```

其中：

- `observation.state[t]` 是 physics step 后读取的真实 `robot.data.joint_pos`，按固定 ABI 重排。
- `action[t]` 是该 step 实际设置给 articulation 的绝对 joint position target，按同一 ABI 重排；不是用 qpos 冒充 action。
- 四路 RGB 来自同一次 `sim.render()` 和同一个 camera tensor batch，具有同一 `frame_id`、仿真时间和单调时钟采集时间。
- recorder 只在新四相机 frame 到达时采样；若 `record_fps < camera_hz`，按仿真时间确定性降采样。
- 图像写入在后台线程执行，physics loop 不等待 MP4/Parquet 编码。队列满时不会堵死仿真，而是增加 `dropped_frames`；正式数据要求该值为 0。

LeRobot v3 的典型目录由官方库生成：

```text
dataset_root/
├── data/
│   └── chunk-.../*.parquet
├── meta/
│   ├── info.json
│   └── ...
└── videos/
    └── observation.images.*/chunk-.../*.mp4
```

LeRobot v2.1 的 shard/chunk 细节可能不同，但同样由对应版本的官方 `LeRobotDataset` 负责，集成层不手工伪造布局。canonical metadata 是 `meta/info.json`。使用 image feature 时可能没有 `videos/` 目录，这是合法的。

对于 ACT、X-VLA、π0/π0.5 和 OpenVLA 类训练：23-D state/action 和四个图像键无需再重排；训练配置仍必须显式选择这些 feature keys、语言 task 和对应 normalization statistics。不同代码库对多相机命名、episode 索引或 action horizon 的适配属于训练入口配置，不应在原始数据里偷偷改语义。

## 10. 验收流程

### 10.1 本地/服务器静态测试

不启动 Isaac Sim：

```bash
python -m unittest discover -s AppleVisionPro/tests -v

python -m py_compile \
  AppleVisionPro/joint_contract.py \
  AppleVisionPro/camera_manager.py \
  AppleVisionPro/lerobot_recorder.py \
  AppleVisionPro/teleop_bridge.py \
  AppleVisionPro/validate_dataset.py \
  AppleVisionPro/smoke_test.py \
  AppleVisionPro/brx_control_server.py \
  scripts/custom/brx_control_server.py
```

这些测试验证固定顺序、左右夹爪物理映射、URDF link/joint/parent/baseline、严格四相机名字和 LeRobot feature schema。

### 10.2 在线只读 smoke test

启动 Isaac Lab 并等到 `Setup complete` 后，在第二个终端执行：

```bash
./isaaclab.sh -p AppleVisionPro/smoke_test.py \
  --url http://127.0.0.1:8765 \
  --timeout_s 60 \
  --save_dir /tmp/brx_four_cameras
```

它不会发送运动命令，会验证：

- 服务 ready；
- live joint names、qpos/action 均为固定 23 维且无 NaN/Inf；
- 可用相机恰好为 `head_left/head_right/left_wrist/right_wrist`；
- 四张图尺寸正确且不是近似常量；
- 左右目图不是完全相同的 attachment。

之后人工查看 `/tmp/brx_four_cameras/*.png`，确认：

- 两个 head view 都覆盖前方桌面、目标物和双臂主要工作区；
- 左右目有合理的小视差，没有对调或一只眼朝后；
- 左腕只跟随物理左手，右腕只跟随物理右手；
- wrist view 能看到目标接近方向和部分夹爪，但夹爪没有遮住大部分画面。

### 10.3 小动作遥操作验收

依次进行，不要直接跳到 1:1 全姿态：

1. bridge 使用 `--scale 0.3 --position_only`，只移动左手约 5 cm，确认物理左臂同方向运动。
2. 只移动右手，确认物理右臂；若错边立即停止，不要在训练数据阶段修补数组。
3. 左右 pinch 分别测试，确认 `pinch↑ → q↑ → 夹爪闭合`。
4. 确认坐标轴后改 `--scale 1.0`。
5. 去掉 `--position_only`，一次只转动一只手 10–20°，确认姿态相对映射。
6. 停止 Vision Pro 发包，确认约 0.35 s 后日志出现 tracking timeout 且机器人 hold。

### 10.4 录制与数据集验收

先录一个 5–10 秒短 episode，停止后查询：

```bash
curl -sS http://127.0.0.1:8765/record/status | python -m json.tool
```

必须检查：

- `episodes_saved >= 1`
- `episode_frames`/`total_frames` 合理
- `dropped_frames == 0`
- `error == null`
- `fps` 等于计划值

完成全部 episodes 后调用一次 finalize，或正常退出 server（退出会自动 save active episode 并 finalize）：

```bash
curl -sS -X POST http://127.0.0.1:8765/record/finalize \
  -H 'Content-Type: application/json' -d '{}'
```

最后运行：

```bash
./isaaclab.sh -p AppleVisionPro/validate_dataset.py \
  datasets/brx_vp_pick_red_001 \
  --repo_id local/brx_vp_pick_red_001
```

默认同时使用安装版本的官方 LeRobot loader 打开首尾 frame，验证四路图像和两个 23-D tensor。仅检查文件/元数据可加 `--skip_loader`，但正式训练前不应跳过 loader 检查。

## 11. WebRTC/CPU/GPU 性能说明

旧控制服务的问题是每个循环都可能 `sim.step(render=True)`，没有墙钟限制，也没有把 viewport render 与训练相机采样分离，因此 CPU 可长期 99%，GPU0 同时承担 Xorg、模型服务和 Isaac Lab 时 WebRTC 更容易卡。

当前循环是：

```text
30 Hz physics: sim.step(render=False)
        ├── 到 render deadline 才 sim.render()        默认 30 Hz
        ├── 到 camera deadline 才复制四路 RGB        默认 15 Hz
        └── 每 step 按 wall clock sleep/catch-up      默认开启
```

PNG 只在有人请求 camera HTTP endpoint 时编码；LeRobot recorder 直接使用同步 raw RGB，不通过四次 HTTP/PNG 往返。

建议调优顺序：

1. 用 `--device cuda:1`、`cuda:2` 或 `cuda:3` 避开 GPU0；不要同时用 `CUDA_VISIBLE_DEVICES` 和错误的逻辑 GPU 编号造成混淆。
2. 先保持 physics 30 Hz，把 `--render_hz` 从 30 降到 20 或 15。
3. 再把 `--camera_hz` 从 15 降到 10，并让 `record_fps <= camera_hz`。
4. 仍不足时用 `--camera_width 512 --camera_height 288`，并让训练配置读取实际 metadata shape。
5. 不要让浏览器或旧 viewer 以高频同时轮询四个 PNG；这会增加 CPU 压缩负载。
6. 观察 `/health` 的 RTF 与 recorder queue，而不只看 `nvidia-smi` 显存。

一个保守低负载配置：

```bash
--sim_dt 0.0333333333 \
--render_hz 20 \
--camera_hz 10 \
--record_fps 10 \
--camera_width 512 \
--camera_height 288
```

如果 physics 本身每 step 已超过 33 ms，wall-clock pacing 不会额外 sleep，CPU 使用率高是算力不足而不是无限循环；此时降低物理复杂度需要单独评估，不能随意改 Isaac Lab core。

## 12. 常见问题与风险

### Vision Pro 能连接但机器人不动

- 检查 `GET :8899/health` 和 `GET :8765/health`。
- 确认双手 `tracking=true`、四元数非零、sequence 递增。
- 确认 bridge 日志已打印 `calibrated`。
- `clutch=false`/`enabled=false` 会故意保持。
- 先用 identity axis map 和 position-only 排除姿态问题。

### 左右手/夹爪反了

不要改 23-D 顺序。检查 Vision Pro 客户端是否把 ARKit left/right hand anchor 发反；服务端物理映射已经按 EE parent links 固定。特别记住 ABI 的右夹爪是 10–11，左夹爪是 19–20。

### 相机黑屏或缺一路

- 必须带 `--enable_cameras`。
- 等待有限 camera warm-up 后再请求。
- 运行 `smoke_test.py` 看 `available` 和 shape。
- 确认使用 `_4cams.urdf`、`--force_usd_conversion` 重新转换过一次且 `merge_fixed_joints=False`。
- 如果 link mode 朝向异常，先用 lookat mode 判断是姿态约定还是渲染问题。

### WebRTC 仍然卡

- 确认 Isaac Lab 不在 GPU0。
- 查看日志实际 render FPS/RTF，而不是把 camera_hz 当成 WebRTC FPS。
- 降低 viewport render_hz；训练相机不需要和 WebRTC 同频。
- 检查是否有旧 `brx_control_server.py` PID 仍在运行或端口占用。

### recorder 开始时报 LeRobot 未安装/格式不匹配

teleoperation 本身不依赖 LeRobot；只有第一次 start episode 才强制创建 dataset。确保安装在 IsaacLab 的 Python 中。`--lerobot_format v2/v3` 必须与安装版本的 dataset codebase format 匹配，不是靠参数完成跨版本转换。

### `dropped_frames > 0`

该 episode 不应直接作为高质量训练数据。先停止录制，降低 record FPS/分辨率、增加 `--record_queue_size` 或解决磁盘/编码瓶颈，然后重新示教。不要用插帧掩盖 action/image 不同步。

### 数据可以打开但训练效果差

结构正确不代表 demonstration 质量足够。检查任务语言一致性、成功/失败 episode 标注策略、相机曝光和遮挡、动作延迟、夹爪闭合语义、episode 起止点、训练代码的 action normalization 和 horizon。首先回放并可视化随机 episode，逐帧确认 `image/state/action` 因果关系。

## 13. 修改原则

后续开发应继续保持：

- 新的 Vision Pro、ROS2 adapter、recorder 或 exporter 只放在 `AppleVisionPro/`。
- 不修改 Isaac Lab core，不把设备特定逻辑写进官方 task/training pipeline。
- `JOINT_NAMES_23` 是不可变 ABI；任何需要新顺序的模型都在训练 adapter 中显式转换。
- 相机严格四路；新增 debug viewport 不得进入 dataset feature schema。
- 原始 state 与真实下发 action 分开记录，不能用 qpos 替代 action。
- 任何 URDF camera pose 修改都要保留 parent link 语义、运行 contract tests、在线 smoke test 和人工视角检查。

