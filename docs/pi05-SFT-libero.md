# PI0.5 SFT、LIBERO 推理与真机动作接口

本文整理 PI0.5 在 LIBERO 上进行推理验证和 SFT 时需要保持的观测、动作、归一化及
checkpoint 契约，并解释这些动作如何连接到 Panda 或 SO-101 真机。

最后更新：2026-09-17。

## 1. 目标与推荐路线

当前工作的核心目标不是直接训练一个长程 checkpoint，而是先建立一条可以判定正确与否的
完整链路：

```text
官方 pi05_libero checkpoint
    ↓
Carrot 原生 LIBERO inference 与 OpenPI 对齐
    ↓
Carrot policy 接入官方 LIBERO evaluator
    ↓
LIBERO 训练数据与推理观测使用同一套变换
    ↓
训练 200 step 并保存 checkpoint
    ↓
加载、推理、resume 和闭环 success rate 对比
```

推荐按以下 gate 顺序推进：

1. **Inference parity**：固定 checkpoint、原始 observation、prompt、noise 和 denoising
   steps，对比 OpenPI PyTorch 与 Carrot 的中间输入和最终 action。
2. **官方 checkpoint 闭环**：用 Carrot inference 驱动官方 LIBERO evaluator，先跑小规模
   smoke，再扩大样本量。
3. **训推预处理一致性**：同一条 LIBERO 样本分别经过训练和推理入口，比较最终模型输入。
4. **200-step SFT**：从 `pi05_base_pytorch` 开始训练，在 step 200 保存。
5. **Checkpoint 闭环**：验证真实加载、固定输入推理以及从 step 200 resume 到 step 201。
6. **效果比较**：用完全相同的 evaluator、任务、初始状态和 seed 比较 base 与 step 200。

200 step 的定位是验证训练、保存、加载和早期学习信号，不足以单独证明已经复现官方
30k-step policy 的最终成功率。

## 2. 已知 OpenPI + LIBERO 基线

使用官方 JAX `pi05_libero` checkpoint 完成了四套正式评测：

| Suite | 成功数 | Success rate |
|---|---:|---:|
| `libero_spatial` | 489/500 | 97.80% |
| `libero_object` | 497/500 | 99.40% |
| `libero_goal` | 487/500 | 97.40% |
| `libero_10` | 465/500 | 93.00% |
| 合计 | 1938/2000 | 96.90% |

评测使用 seed 7、每个 task 50 episodes、每个 suite 10 tasks、`replan_steps=5`。完整记录见
[`docs/test-manifest/openpi-libero-official-benchmark-20260916.md`](test-manifest/openpi-libero-official-benchmark-20260916.md)。

同一 JAX checkpoint 转换成 OpenPI PyTorch checkpoint 后，JAX 和 PyTorch eager policy 在
`libero_spatial` 的相同 10 个 smoke episodes 上均为 10/10。默认
`torch.compile(mode="max-autotune")` 的首次编译曾超过 websocket keepalive；eager 路径可正常
完成评测。记录见
[`docs/test-manifest/openpi-libero-jax-pytorch-smoke-20260917.md`](test-manifest/openpi-libero-jax-pytorch-smoke-20260917.md)。

在 H20 上必须将 policy 与 EGL renderer 放在不同物理 GPU，并保证每张物理 GPU 最多一个
LIBERO renderer。曾经将 PyTorch policy 与 EGL renderer 放在同一 H20 上，触发 NVIDIA
Xid 31 和 Xid 109；这不是普通性能问题，而是正式评测的安全约束。

## 3. LIBERO-Panda observation contract

LIBERO 中的机器人是 Panda，但 policy 接收的不是七个机械臂关节角，而是末端执行器状态。

### 3.1 状态定义

本节先定义 policy observation 中与机器人状态有关的量。

| 符号 | 形状 / 类型 | 作用域 | 含义 |
|---|---|---|---|
| $p$ | `[3]` float vector | LIBERO observation | 末端执行器位置 $[x,y,z]$ |
| $r$ | `[3]` float vector | LIBERO observation | 末端姿态的 axis-angle 表示 |
| $g$ | `[2]` float vector | LIBERO observation | 两个夹爪手指关节的位置 |
| $s$ | `[8]` float vector | LIBERO observation | 拼接后的 policy state |
| $s_{32}$ | `[32]` float vector | PI0.5 model input | 将 $s$ 补零后的模型状态 |

LIBERO evaluator 从仿真器读取末端位置、末端四元数和两个夹爪关节位置。四元数先转换成
axis-angle，因此 policy state 为：

$$
s = [p,r,g]
  = [x,y,z,r_x,r_y,r_z,g_1,g_2]
  \in \mathbb{R}^{8}.
$$

PI0.5 使用固定的 `action_dim=32` 统一不同机器人，因此状态随后补零：

$$
s_{32} = [s,0,\ldots,0] \in \mathbb{R}^{32}.
$$

后面的 24 维只是 padding，不代表 Panda 有 32 个状态自由度。Panda 实际有七个机械臂
旋转关节；LIBERO policy 选择在末端操作空间观察和控制，因此没有把
`robot0_joint_pos` 放进 policy state。

### 3.2 图像定义

本节定义 policy 使用的相机槽位。image mask 是“这一路图像是否真实存在”的标志，不是
逐像素分割 mask。

| 模型槽位 | LIBERO 来源 | Mask | 含义 |
|---|---|---:|---|
| `base_0_rgb` | `agentview_image` | `True` | 环境第三人称相机 |
| `left_wrist_0_rgb` | `robot0_eye_in_hand_image` | `True` | Panda 手腕相机 |
| `right_wrist_0_rgb` | 全零图像 | `False` | 补齐 PI0.5 的固定三相机接口 |

官方 evaluator 会把两幅 LIBERO 图像旋转 180 度，再使用 padding resize 到 `224×224`。
训练和推理必须采用相同方向、通道顺序、数值范围和 resize 规则。

### 3.3 Prompt

每个 LIBERO task 都有自然语言描述。训练时 `prompt_from_task=True`，推理时 evaluator 将
相同语义的 task description 放入 `prompt`。prompt 不是可选装饰：tokenizer、padding
方向、最大 token 长度和 attention mask 都属于训推一致性的一部分。

## 4. LIBERO-Panda action contract

### 4.1 单步动作

LIBERO 使用末端操作空间动作，而不是七个 Panda 关节角。

| 符号 | 形状 / 类型 | 作用域 | 含义 |
|---|---|---|---|
| $\Delta p_t$ | `[3]` float vector | step $t$ | 末端位置增量 |
| $\Delta r_t$ | `[3]` float vector | step $t$ | 末端姿态增量 |
| $u_{g,t}$ | 标量 | step $t$ | 夹爪开合命令 |
| $a_t$ | `[7]` float vector | step $t$ | LIBERO 单步动作 |
| $a_{t,32}$ | `[32]` float vector | PI0.5 target | 补零后的模型动作 |

单步动作定义为：

$$
a_t = [\Delta p_t,\Delta r_t,u_{g,t}]
    = [\Delta x_t,\Delta y_t,\Delta z_t,
       \Delta r_{x,t},\Delta r_{y,t},\Delta r_{z,t},u_{g,t}]
    \in \mathbb{R}^{7}.
$$

前六维描述相对当前末端位姿的变化，最后一维控制夹爪。LIBERO 数据中的 action 已经是
delta action，因此官方 `pi05_libero` 使用 `extra_delta_transform=False`。再次减去当前 state
会重复做 delta 转换，改变监督目标。

为了进入固定的 PI0.5 action head，训练目标补到 32 维：

$$
a_{t,32} = [a_t,0,\ldots,0] \in \mathbb{R}^{32}.
$$

推理输出反归一化后只保留前七维交给 LIBERO。

### 4.2 Action chunk 与 replanning

本节引入 action horizon，说明“一次预测十步”和“实际执行五步”为什么不矛盾。

| 符号 | 形状 / 类型 | 作用域 | 含义 |
|---|---|---|---|
| $H$ | 标量整数，$H=10$ | PI0.5 LIBERO | action horizon |
| $K$ | 标量整数，$K=5$ | 官方 evaluator | 每次重新规划前执行的动作数 |
| $A_t$ | `[H, 7]` float matrix | time $t$ | 从当前观测预测的 action chunk |

PI0.5 每次预测未来十步：

$$
A_t = [a_t,a_{t+1},\ldots,a_{t+H-1}]
    \in \mathbb{R}^{H\times 7},\qquad H=10.
$$

官方 evaluator 只执行其中前 $K=5$ 步，然后获取新 observation 并重新预测。也就是说：

```text
观察环境 → 预测 10 步 → 执行 5 步 → 重新观察 → 再预测 10 步
```

`action_horizon` 决定模型预测多远，`replan_steps` 决定开环执行多久。为缩短评测而改变
二者会改变 policy 行为，不能视为等价加速。

## 5. Normalization contract

状态和动作的各维量纲、范围不同，训练前必须归一化；推理输出必须使用同一份统计量进行
逆变换。统计量是 checkpoint 行为的一部分，而不只是数据集元信息。

### 5.1 官方 PI0.5 LIBERO 使用 mean/std

| 符号 | 形状 / 类型 | 作用域 | 含义 |
|---|---|---|---|
| $x$ | 标量 | 某个 state/action 维度 | 原始物理量 |
| $\mu$ | 标量 | 同一维度 | 训练集均值 |
| $\sigma$ | 非负标量 | 同一维度 | 训练集标准差 |
| $\epsilon$ | 标量，$10^{-6}$ | 数值稳定项 | 避免除零 |
| $x_{\mathrm{norm}}$ | 标量 | model space | 归一化后的数值 |

训练和推理输入使用 z-score：

$$
x_{\mathrm{norm}} = \frac{x-\mu}{\sigma+\epsilon}.
$$

模型输出 action 后执行逆变换：

$$
x = x_{\mathrm{norm}}(\sigma+\epsilon)+\mu.
$$

官方 OpenPI 配置对 PI0/PI0.5 使用 mean/std normalization。Carrot 当前 RoboTwin inference
使用的是 `q01/q99` quantile normalization，因此不能直接复用于 `pi05_libero`。

### 5.2 为什么不能混用 stats

如果模型在训练时学习的是 $x_{\mathrm{norm}}$，推理却使用另一机器人的统计量
$\mu',\sigma'$，则实际解码出的动作变成：

$$
x' = x_{\mathrm{norm}}(\sigma'+\epsilon)+\mu'.
$$

即使模型权重和归一化空间里的预测完全正确，$x'$ 也不再等于训练语义下的 $x$。因此
checkpoint 加载成功、输出有限数值或 tensor shape 正确，都不能替代 stats 一致性验证。

## 6. 训推一致性

训练与推理的原始输入来源不同，但进入模型前的语义必须相同。

### 6.1 训练路径

```text
physical-intelligence/libero 样本
→ 将 dataset keys repack 成 image / wrist_image / state / actions / prompt
→ LIBERO-Panda input transform
→ mean/std normalization
→ 图像 resize、tokenize、state/action padding
→ PI0.5 loss
```

### 6.2 推理路径

```text
LIBERO simulator observation
→ 图像旋转 180 度并 resize
→ 构造 8 维末端 state 和 prompt
→ 同一个 LIBERO-Panda input transform
→ 同一份 mean/std normalization
→ 同样的图像、token 和 state padding
→ PI0.5 sampling
→ action unnormalize
→ 取前 7 维
→ LIBERO env.step
```

### 6.3 必须固定的 parity 输入

为了区分模型误差与随机采样误差，OpenPI PyTorch 和 Carrot policy comparison 必须固定：

- 同一个由官方 `pi05_libero` 转换的 PyTorch checkpoint；
- 同一条原始 observation 和 prompt；
- 同一份 checkpoint normalization stats；
- 同一个 sampling noise tensor；
- 相同 denoising steps；
- 相同 image resize、dtype、范围和通道顺序。

比较项目至少包括：

- 转换后的三路图像及 image masks；
- normalized state；
- prompt token ids 和 token mask；
- model-space raw actions；
- unnormalize 后的 `[10, 7]` actions。

数值报告应同时包含 FP64 Dice distance、absolute/reference-relative P50/P90/P99/Max，并保留
严格 pointwise assertion。Dice 只能作为补充相似度指标，不能替代逐点正确性。

## 7. Carrot 当前 inference 的边界

当前 `create_trained_policy` 和 `Pi05Policy` 是 RoboTwin 专用路径，假设：

- 14 维 state；
- 三路真实 RGB CHW 相机；
- RoboTwin joint/gripper preprocessing；
- 14 维 action decoding；
- `q01/q99` quantile normalization；
- action horizon 通常为 50。

这套路径不能直接加载 `pi05_libero` 后声称推理正确。建议增加独立的 LIBERO policy adapter，
保持现有 RoboTwin 路径不变。命名应使用 `LiberoInputs`、`LiberoOutputs` 或
`LiberoPolicyAdapter`，因为被适配的是完整 LIBERO-Panda contract，而不只是 Panda 机械结构。

在未来出现多个 Panda 环境且它们确实共享 state/action contract 后，再考虑抽象通用 Panda
embodiment 层。

## 8. LIBERO SFT 配置基线

官方 OpenPI `pi05_libero` 配置的关键项为：

| 配置 | 官方值 | 语义 |
|---|---:|---|
| Dataset | `physical-intelligence/libero` | LIBERO LeRobot 数据 |
| `prompt_from_task` | `True` | 使用 task description |
| `action_horizon` | 10 | 每次监督十步 action |
| `discrete_state_input` | `False` | state 不离散编码进 prompt |
| `extra_delta_transform` | `False` | 数据本身已是 delta action |
| Global batch size | 256 | 官方训练批大小 |
| Warmup | 10,000 steps | 官方 LR warmup |
| Peak LR | $5\times10^{-5}$ | 官方 peak learning rate |
| EMA decay | 0.999 | 官方参数 EMA |
| Training steps | 30,000 | 官方完整训练长度 |

Carrot 的短实验不能不加说明地套用原 RoboTwin 配置。当前需要显式决定并记录：

- 使用官方 batch 256，还是为了现有集群吞吐使用其他 global batch；
- 200-step run 是否沿用 10k warmup；若沿用，它几乎全部处于 warmup 早期；
- Carrot 当前没有 EMA 路径，与官方 `ema_decay=0.999` 存在差异；
- norm stats 是直接复用官方 checkpoint assets，还是由完全相同 transform 重新计算并验证；
- checkpoint 评测使用 raw training weights，还是未来补齐 EMA weights。

短实验的第一目标应是证明 Carrot 能使用正确数据 contract 学习、保存并推理，而不是在
200 step 内追求官方最终 success rate。

## 9. 200-step checkpoint 实验

建议在前述 inference 和数据 parity gate 通过后启动独立实验，不覆盖正式配置或旧输出。

### 9.1 训练设置

- 初始化 checkpoint：`pi05_base_pytorch`；
- 数据：`physical-intelligence/libero`；
- action horizon：10；
- 独立 `output_dir` 和 W&B run name；
- `steps=200`；
- `save_freq=200`；
- 所有数据 contract 和超参数写入实验记录。

### 9.2 Checkpoint 验收

step 200 后至少检查：

1. 根目录存在 `model.safetensors`、`config.json`、tokenizer 与 normalization artifacts；
2. optimizer DCP 和 `trainer_state.json` 完整；
3. Carrot LIBERO policy 能真实加载，输出有限的 `[10, 7]` actions；
4. 固定 observation、prompt 和 noise 时输出可重复；
5. 能从 step 200 恢复 optimizer、scheduler 和 step，并继续完成 step 201。

### 9.3 改善判据

仅看到 train loss 下降不能证明 policy 改善。base 与 step 200 必须使用：

- 相同 policy 实现；
- 相同 checkpoint assets 和 inference contract；
- 相同任务集合；
- 相同初始状态与 seed；
- 相同 `replan_steps` 和 episode step limit。

结果至少报告固定 held-out 数据上的 loss，以及同一小规模闭环集上的 success count。小样本
success rate 噪声很大，只用于方向判断；正式结论仍需更大 episode 数。

## 10. 从 PI0.5 action 到真机

PI0.5 不负责自动识别机器人的硬件接口。模型输出的语义由训练数据决定，robot adapter 和
controller 负责将它转换为硬件命令。

```text
PI0.5 action
→ checkpoint-matched unnormalize
→ robot-specific action adapter
→ Cartesian controller / IK / joint mapping
→ safety filter 与轨迹插值
→ 真机驱动器
```

### 10.1 Panda

Franka Panda 有七个机械臂旋转关节。Franka Control Interface 同时支持关节位置、关节速度、
关节力矩、Cartesian pose 和 Cartesian velocity，因此 Panda 真机并非只能接收关节角。

#### 路径 A：Cartesian controller

本节定义如何把 LIBERO 风格的末端增量累积为目标位姿。

| 符号 | 形状 / 类型 | 作用域 | 含义 |
|---|---|---|---|
| $T_t$ | $4\times4$ homogeneous transform | time $t$ | 当前末端位姿 |
| $\Delta\xi_t$ | `[6]` twist vector | time $t$ | PI0.5 输出的末端位姿增量 |
| $\widehat{\Delta\xi_t}$ | $4\times4$ Lie algebra matrix | time $t$ | 将 twist 写成矩阵形式 |
| $T_{\mathrm{target}}$ | $4\times4$ homogeneous transform | time $t$ | 控制器要跟踪的目标位姿 |

如果 action 定义在末端局部坐标系，可写为：

$$
T_{\mathrm{target}}
= T_t\operatorname{Exp}(\widehat{\Delta\xi_t}).
$$

若数据将增量定义在 base/world frame，则乘法顺序和旋转处理会不同。坐标系必须由训练数据
contract 明确定义，不能凭经验猜测。目标位姿还应经过工作空间限制和平滑插值，再交给
Franka Cartesian controller。

#### 路径 B：受约束 IK

如果下游接口只接受关节目标，需要求一个靠近当前姿态且满足目标末端位姿的关节解。

| 符号 | 形状 / 类型 | 作用域 | 含义 |
|---|---|---|---|
| $q_t$ | `[7]` joint vector | time $t$ | 当前 Panda 关节角 |
| $q$ | `[7]` optimization variable | IK | 待求关节角 |
| $f(q)$ | SE(3) pose | IK | Panda 正运动学 |
| $W$ | positive weight matrix | IK | 位置和姿态误差权重 |
| $\lambda$ | 非负标量 | IK | 偏好靠近当前姿态的正则强度 |
| $q^*$ | `[7]` joint vector | IK output | 要发送的目标关节角 |

一个基本目标可写为：

$$
q^* = \underset{q}{\arg\min}\;
\left\|f(q)-T_{\mathrm{target}}\right\|_W^2
+ \lambda\left\|q-q_t\right\|^2.
$$

实际求解还必须加入关节限位、速度和加速度限制、自碰撞、环境碰撞及奇异位形约束。多个
IK 解存在时，正则项用于偏好当前关节姿态附近的连续解。

无论采用哪条路径，PI0.5 action chunk 都不能直接以低频、离散跳变的形式写入 Panda 实时
控制环；中间需要轨迹生成和高频跟踪层。

### 10.2 SO-101

SO-101 的 LeRobot follower driver 默认接收具名的电机目标位置，例如 shoulder、elbow、
wrist 和 gripper 的 `.pos`，然后写入舵机 `Goal_Position`。因此推荐直接在 SO-101 数据上
训练 joint-position action：

$$
a_{\pi}
\xrightarrow{\mathrm{unnormalize}}
q_{\mathrm{target}}
\xrightarrow{\mathrm{safety}}
q_{\mathrm{safe}}
\longrightarrow
\mathrm{send\_action}.
$$

其中 $a_{\pi}$ 是 PI0.5 的归一化输出，$q_{\mathrm{target}}$ 是校准后的舵机目标位置，
$q_{\mathrm{safe}}$ 是经过单步变化、关节限位和速度限制后的最终命令。

LeRobot 也提供从末端位姿到关节位置的 IK processor，但 SO-101 机械臂部分只有五个自由度，
不能任意满足六维末端位姿。其官方 IK processor 因此允许降低 orientation weight，让位置
优先而姿态只软约束。由此可见，`pi05_libero` 的 Panda 六维末端动作不能仅通过一个通用
IK adapter 就可靠迁移到 SO-101。

SO-101 的推荐路线是：

```text
pi05_base
→ 使用已校准 SO-101 采集的数据微调
→ 直接预测 SO-101 joint targets
→ 限速、限位和 watchdog
→ LeRobot send_action
```

### 10.3 真机安全层

模型原始输出不能直接写入电机。最低限度需要：

- NaN/Inf 与 shape 检查；
- 单步最大末端位移、旋转和关节变化；
- 关节位置、速度和加速度限制；
- 工作空间、自碰撞和环境碰撞检查；
- IK 无解或接近奇异位形时拒绝动作；
- action timeout 和 watchdog；
- 急停与人工接管；
- 首次测试使用低速度、空工作区和短 action chunk。

## 11. 当前结论

1. `pi05_libero` 的七维输出是六维末端 delta pose 加一维夹爪命令，不是 Panda 七个关节角。
2. LIBERO policy state 是六维末端 pose 加两个夹爪位置，共八维；Panda 七个机械臂关节仍由
   仿真或真机底层 controller 控制。
3. Carrot 必须先增加独立 LIBERO policy adapter，并用官方 checkpoint 验证完整 inference
   parity 和闭环 success，才能开始解释自训 checkpoint 的结果。
4. LIBERO SFT 必须使用 LIBERO 数据、LIBERO mean/std stats、10-step action chunk 和相同
   prompt/image/state/action transforms。
5. 200-step checkpoint 值得做，但它首先是训练、导出、加载、resume 和早期学习信号测试。
6. Panda 可以使用 Cartesian controller 或受约束 IK；SO-101 更适合直接学习校准后的关节
   目标位置。不同机器人不能只替换一个输出维度就共享 policy action contract。

## 12. 参考资料

- [OpenPI LIBERO benchmark README](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/README.md)
- [OpenPI `pi05_libero` training config](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/training/config.py)
- [OpenPI LIBERO policy transforms](https://github.com/Physical-Intelligence/openpi/blob/main/src/openpi/policies/libero_policy.py)
- [OpenPI LIBERO evaluator](https://github.com/Physical-Intelligence/openpi/blob/main/examples/libero/main.py)
- [Franka `Robot` control interfaces](https://frankarobotics.github.io/libfranka/latest/classfranka_1_1Robot.html)
- [Franka Control Interface overview](https://frankarobotics.github.io/docs/doc/libfranka/docs/overview.html)
- [LeRobot SO-101 setup and calibration](https://huggingface.co/docs/lerobot/main/en/so101)
- [LeRobot SO follower implementation](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/so_follower.py)
- [LeRobot SO-101 kinematic processors](https://github.com/huggingface/lerobot/blob/main/src/lerobot/robots/so_follower/robot_kinematic_processor.py)
