# LIBERO 全任务目录与真实 RL 验收

状态：131 个原生任务定义已完成，全量 CPU / GPU 验收待运行。按用户要求先提交代码，随后测试，修复另作提交。
分支 `nrwu/newLiberoWithArena`；Python 3.12 / Isaac Sim 6 / IsaacLab 3 / Arena 沿用项目锁定版本。
历史首批 5 任务已在提交 `cec1e83` 通过真实 GPU / PPO 验收；该结果不代表本次 131 任务全部通过。

## 任务范围

每个 case 有独立 Python 模块，显式定义语言、资产、布局、关节初态和成功条件。

| suite | case 数 |
| --- | ---: |
| libero_spatial | 10 |
| libero_object | 10 |
| libero_goal | 11 |
| libero_10 | 10 |
| libero_90 | 90 |

来源为 LW-BenchHub `b2bcb2d00edef691f9fcc49039cbf0bcc7464605` 的
`lw_benchhub_tasks/lightwheel_libero_tasks`；`task_sources.json` 保存全部源文件、注册类和 Gym ID。
上游一个 L10L2 文件注册三个 case，本实现拆成三个模块；L10K6 按实际注册类和任务语义命名，修正上游误导性的文件名。
运行时不导入 LW-BenchHub 或 MuJoCo。注册表可在未启动 Isaac 时读取。

采用固定厨房桌面布局，不声称复现 LW 的房间布局、随机化分布或原始 LIBERO 轨迹。
成功条件逐环境计算，多目标条件取 AND，目标物体身份明确；包含真实接触、释放、稳定、容器边界和关节状态检查。
源代码的 env 0 广播、宽泛任意物体匹配和 caddy 分区 TODO 按任务语义实现。
长柄锅在架子内采用部分插入条件；BBQ sauce 使用原始 LIBERO 资产转换，属于有意的资产差异。
critic 维数随物体和关节数变化，请查询 `env.single_observation_space["critic"].shape`。

## 验收与执行

每个任务检查真实 GPU reset/step、两路 RGB、部分 reset 隔离、超时 bootstrap、成功终止与 PPO 参数更新。
合成成功 fixture 验证物理条件和奖励接口，不代表策略已经学会任务。

按当前 Gemini 会话确定 MY_DFS，确认 GPU 无其他 renderer，暂停并在结束后恢复 dguard。
H20 每块物理 GPU 最多一个 renderer；当前脚本逐任务串行运行，不停止既有 Ray head。

```bash
export ARENA_LIBERO_ASSETS=/root/arena-libero-all-assets-v2
export ARENA_LIBERO_OUTPUT="${MY_DFS}/benchmarks/arena-libero-rl/<new-run>"
bash tests/arena_libero/test_task_catalog.sh
```

资产准备先完成 SDK 对象缓存下载，再转换原始 LIBERO BBQ sauce，最后创建新的资产目录：

```bash
python scripts/arena_libero/convert_libero_object.py \
  --source /path/to/LIBERO/libero/libero/assets/stable_hope_objects/bbq_sauce/bbq_sauce.xml \
  --output /root/arena-libero-converted-assets-v2/LiberoBbqSauce.usd
python scripts/arena_libero/prepare_assets.py \
  --scene /root/.cache/lightwheel_sdk/floorplan/robocasa-libero-1-1/scene_enabled.usd \
  --object-cache /root/.cache/lightwheel_sdk/object \
  --libero-bbq-usd /root/arena-libero-converted-assets-v2/LiberoBbqSauce.usd \
  --output /root/arena-libero-all-assets-v2
```

离线转换使用 MuJoCo 读取质量、质心和惯性，生成原生 USD、贴图和碰撞体；不增加模拟器运行时依赖。
wrapper 仍引用 SDK 缓存，缓存和原场景必须保持可访问；脚本拒绝覆盖已有输出。

## 选择任务

```python
from pathlib import Path
from carrot_sim.arena_libero import ArenaLiberoConfig
from carrot_sim.arena_libero.tasks import list_tasks

for task in list_tasks("libero_spatial"):
    print(task.task_id, task.language)

config = ArenaLiberoConfig(
    asset_root=Path("/root/arena-libero-all-assets-v2"),
    task_id="libero_10/L10K4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    num_envs=4,
)
# AppLauncher 启动后导入 make_env，再调用 make_env(config)。
```

`task_id=None` 保留原两物体 smoke 环境。`env.task_description` 提供语言，`env.task_name` 为 suite/name。
一个 batch 使用同一任务，reset 不切换任务。

## 扩展运行时的已有抽样记录

`20260922-all-runtime-01` 的 spatial on-stove case 已通过 4 环境、两路 RGB、128 transitions、
2 次 PPO 更新、部分 reset 和物理成功 fixture。证据位于
`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/benchmarks/arena-libero-rl/20260922-all-runtime-01/`。
它早于最终全量源码，仍需回归；下文仅保留首批 5 任务历史记录。

## 2026-09-22 CPU

执行代理在远端 `/opt/venvs/carrot` Python 3.12 下运行旧接口测试与新增任务目录/谓词测试：
16 passed（1.99s）。GPU 任务检查尚未运行，不沿用此前单任务 PASS。

## 首批 5 任务历史资产准备

已下载对象缓存：Bowl008、Plate012、Bowl009、Cookies002、Bottle054、Pot086。
下载使用 `lightwheel_sdk.loader.object_loader.acquire_by_registry`，类型为 `objects` / `USD`；
仅准备资产时使用 SDK，运行环境不导入 LW-BenchHub。

以下是首批 5 任务旧版本使用的命令；当前版本请使用上文含 BBQ 参数的命令：

```bash
python scripts/arena_libero/prepare_assets.py \
  --scene /root/.cache/lightwheel_sdk/floorplan/robocasa-libero-1-1/scene_enabled.usd \
  --object-cache /root/.cache/lightwheel_sdk/object \
  --output /root/arena-libero-five-assets-v2
```

木柜、炉灶和摩卡壶分别来自 source scene 的
`/world/storage_furniture_right_group_1`、`/world/stovetop_front_group_1`、
`/world/mokapot_1_front_group_1`。提取保留引用与内部关节，去掉根的原场景变换并归零世界固定
关节锚点；任务配置显式恢复测量所得缩放与朝向，再给定新布局的位置。
原始 scene 和所有引用文件仍必须可访问，wrapper 不是脱离缓存的自包含资产包。

木柜三个滑动关节从上到下为 `StorageFurniture136_Drawer001_joint` 至 `003_joint`；
炉灶旋钮为 `knob_center_joint`，摩卡壶的自由根保留被动 `MokaPot001_Lid_joint`。
炉灶“打开”由真实旋钮角度判定，不包含热传导仿真。

静态源资产记录：
`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/benchmarks/arena-libero-five-assets/inventory_20260922.json`。
该记录是 v1 wrapper 的盘点，运行验收另记录实际资产目录。

## 2026-09-22 five-01：FAIL

首个 between 任务在创建 termination 配置时失败，未进入任务步：IsaacLab 的配置字符串
包装器会递归写入参数，而传入的 GoalSpec 是 frozen dataclass，触发 FrozenInstanceError。
已改为在 ManagerTermCfg 边界传递普通字典；目录定义本身仍保持不可变。
没有修改第三方源码或降级依赖。

任务 `5f3f1fce-0736` 的日志保存在
`${MY_DFS}/benchmarks/arena-libero-rl/20260922-five-01/libero_spatial/LS_pick_up_black_bowl_between_plate_and_ramekin_and_place_it_on_plate/runner.stdout`。
Kit fast shutdown 可能使进程返回码掩盖 Python traceback；因此脚本还要求每个任务写出
完整 result.json，缺少结果立即失败，最终再独立校验全部产物。
该轮没有 result.json 或图像，不计作 GPU 通过；执行代理确认已恢复 dguard、无残留 renderer。

## 2026-09-22 five-02：3 PASS，整轮 FAIL

输出根 `${MY_DFS}/benchmarks/arena-libero-rl/20260922-five-02`，实际 GPU 执行任务
`5f3f1fce-0745`。此前 `-0743` 仅因 MY_DFS 未 export 退出，没有启动 GPU。

三个 spatial 任务各完成 4×32 transitions、8 次超时和两次 PPO 参数更新；成功 fixture、
部分 reset、terminal observation 均通过。主模型独立读取结果并查看外部相机图像：
两只黑碗与干扰物清晰可见，顶抽屉打开且目标碗实际位于其中。

第 4 个炉灶任务失败：IsaacLab 将选中全部关节的 SceneEntityCfg.joint_ids 优化成 slice，
原实现按 list 取第 0 个元素引发 TypeError。已改为先按 selector 切 tensor，再取单个关节。
底抽屉任务本轮未运行。新增可移动 articulation 的 is_fixed_base=False 断言检查摩卡壶自由根。
执行代理确认已清理 renderer、恢复 dguard；原 Ray head 保留。


## 2026-09-22 five-03：最终 PASS

执行任务 `5f3f1fce-0753`，资产目录 `/root/arena-libero-five-assets-v2`。
完整证据目录：
`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/benchmarks/arena-libero-rl/20260922-five-03/`。

每个任务单独启动一个 Kit 进程、4 个环境、GPU 0 一个 renderer，顺序执行。
5 个任务分别完成 128 transitions（合计 640）和两次 PPO actor/critic 参数更新；
每项成功 fixture、部分 reset、terminal observation、纯超时 bootstrap 均 PASS。
摩卡壶额外验证为自由根 articulation；顶抽屉目标初态检查实际位于打开的上层内部。

| 任务 | critic 维数 | actor 最大更新 | critic 最大更新 |
| --- | ---: | ---: | ---: |
| L10K3_turn_on_the_stove_and_put_the_moka_pot_on_it | 69 | 0.0006004 | 0.0005946 |
| L10K4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it | 71 | 0.0006004 | 0.0006002 |
| LS_pick_up_black_bowl_between_plate_and_ramekin_and_place_it_on_plate | 91 | 0.0006004 | 0.0005950 |
| LS_pick_up_black_bowl_in_top_drawer_of_wooden_cabinet_and_place_it_on_plate | 110 | 0.0006004 | 0.0005864 |
| LS_pick_up_the_black_bowl_from_table_center_and_place_it_on_the_plate | 91 | 0.0006004 | 0.0006004 |

实际版本：Python 3.12.13、Isaac Sim 6.0.1.0、IsaacLab 3.0.0b2.post1、
Arena 0.3.0、Torch 2.11.0+cu128。Docker image tag 未记录。
主模型独立读取全部结果、检查两路相机图像，并确认本地与远端 32 个实现/测试/脚本文件
SHA256 相同；`lw_imported=false`。这些结果只证明任务与 RL 链路可运行，不代表 PPO 已学会任务。

完整 shell exit 0，独立 validator 输出 `Five task artifacts PASS`。
执行代理收尾确认无残留 Kit、无 Xid；既有 Ray head 保留、GPU 资源使用 0/8；
dguard 已恢复，恢复定时器已取消。完整 stdout 保存在证据目录根下。

## 2026-09-22 all-committed-01：提交后验证进行中

被测提交 `68cc7673488e9a7d2e0f98adba3e92b119117ba9`。本地 Python 3.12 CPU：17 passed / 1.35s；
远端 `/opt/venvs/carrot`：17 passed / 2.05s。checksum 同步无差异，170 行源码 manifest 的 SHA256 为
`3b05b2a7bfc706c4d6f8970cb984da610889b0b344132c9d51261f13785647a6`。

BBQ v2 USD 与原 MJCF 的独立比较 PASS：质量误差 4.24e-10 kg、质心最大误差 3.72e-10 m、
对角惯量最大误差 5.05e-13 kg·m²；三个碰撞体及贴图引用检查通过。
证据根目录：`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/benchmarks/arena-libero-rl/20260922-all-committed-01/`，
包含 `cpu.stdout`、`source-files.sha256`、`source-commit.txt`、`bbq_mass_comparison.json`。
本小节尚不代表 GPU 或全任务通过；Docker image tag 未记录。

同一提交的 LO BBQ 入篮任务真实 GPU PASS（Gemini job `5f3f1fce-0857`，exit 0）：
4 环境、critic 121 维、128 transitions、8 truncations，actor 更新 0.0006004013、
critic 更新 0.0005775355；两路 RGB、部分 reset、terminal observation、物理成功 fixture 全部通过。
版本为 Python 3.12.13 / Isaac Sim 6.0.1.0 / IsaacLab 3.0.0b2.post1 / Arena 0.3.0 / Torch 2.11.0+cu128。
该任务 `result.json`、`initial_state.json` 和两张图像位于上述证据根目录的
`libero_object/LO_pick_up_the_bbq_sauce_and_place_it_in_the_basket/`。这只是 131 个 case 中的一个。

GPU 映射已确认：该任务 `runner.stdout` 的 Vulkan 表仅物理 GPU 1 标记 `Active=Yes:0`，
实际 `Environment device : cuda:1`；GPU 1 承担约 5.8 GiB 显存和主要利用率。
构建时较早打印的 `SimulationCfg(device='cuda:0')` 是 `environment.py` 设置 device 之前的配置，
不代表实际运行设备。其余卡的小 CUDA 上下文不作为多卡 renderer 的证据。

## 2026-09-22 all-committed-02 / 03：代表任务与失败定位

02 在 `68cc767` 上运行 9 个代表任务：5 PASS（酒架、双物体入篮、锅在架下、叠碗入托盘、书在 caddy 前格），
4 FAIL（沙拉酱入篮、书在 caddy 后格、微波炉、锅在架中层）。每个 PASS 都有 128 transitions 和非零 PPO 参数更新。
03 使用 `6b4174f` 诊断测试代码，生产源码未变，复现全部 4 个失败。两轮证据位于相邻的
`20260922-all-committed-02/`、`20260922-all-committed-03/`，失败日志在 `logs/`。

沙拉酱和后格书的容器、稳定、接触均满足，TCP 距离分别 0.3383 / 0.3233 m，未满足原任务 0.35 m 阈值。
微波炉门已关闭，但 TCP 距离 0.3904 m 未达到 0.4 m，且杯完整几何底界比原生容器下界低 1.2875 mm。
修正夹具：摆物体之前让 env 0 的手通过真实控制向基座退让并抬高；保持所有释放阈值。
微波炉内腔只将底界从 -0.070 移至 -0.072 m，容纳完整视觉/碰撞几何与支撑面的偏差；上界及 XY 不变。
锅在架中层最终掉到桌面，接触为 0；继续记录沉降轨迹定位，不放宽生产成功条件。以上修正仍待 GPU 重跑。
