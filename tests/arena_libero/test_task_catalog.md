# 五任务目录与真实 RL 验收

状态：PASS（2026-09-22 five-03）。5 个任务真实 GPU / PPO 验收通过，CPU 16 项通过。基线 `dea550f`，分支 `nrwu/newLiberoWithArena`。
Docker image tag 未记录；依赖沿用项目 Python 3.12 / Isaac Sim 6 / IsaacLab 3 / Arena 锁定版本。

任务目录：`src/carrot_sim/arena_libero/tasks/libero_spatial/`（3 个任务）和
`src/carrot_sim/arena_libero/tasks/libero_10/`（2 个任务）。任务文件各自定义语言、物体、
初始位置、关节状态和成功条件；注册表不依赖 Isaac 或 LW 的 Python 包。

参考 LW-BenchHub 同名任务的语义和资产，不声称复现其所有厨房布局、随机化分布或原始
MuJoCo LIBERO 初态。当前使用一个固定厨房桌面，多个任务的 critic 维数按物体和关节数量
确定；通过 `env.single_observation_space["critic"].shape` 查询，不能硬编码 52。

## 预期检查

- 五个独立文件和可选择的 task ID，两个黑碗的目标身份明确。
- 三种成功判据按 batch 逐行计算；拒绝悬空、未释放、抽屉未关、碗越界或炉灶未开。
- 每个任务真实 GPU reset/step、两路 RGB、部分 reset 隔离和 PPO 参数更新。
- 抽屉和旋钮使用真实关节，Panda 通过接触操纵；reset 恢复关节初态。
- 合成成功 fixture 验证奖励/终止的物理条件，不等同于策略学会任务。

## 执行

按当前 Gemini 会话确定 MY_DFS，确认 GPU 0 无其他 renderer，暂停并在结束后恢复 dguard。
资产准备独立于测试；`ARENA_LIBERO_ASSETS` 内需要各任务引用的 USD 及其依赖。

```bash
export ARENA_LIBERO_ASSETS=/root/arena-libero-five-assets-v2
export ARENA_LIBERO_OUTPUT="${MY_DFS}/benchmarks/arena-libero-rl/<five-task-run>"
bash tests/arena_libero/test_task_catalog.sh
```

GPU 每次只启动一个进程/renderer，任务顺序运行；不改依赖，不停止既有 Ray head。

## 选择任务

```python
from pathlib import Path
from carrot_sim.arena_libero import ArenaLiberoConfig
from carrot_sim.arena_libero.tasks import list_tasks

for task in list_tasks("libero_spatial"):
    print(task.task_id, task.language)

config = ArenaLiberoConfig(
    asset_root=Path("/root/arena-libero-five-assets"),
    task_id="libero_10/L10K4_put_the_black_bowl_in_the_bottom_drawer_of_the_cabinet_and_close_it",
    num_envs=4,
)
# AppLauncher 启动后导入 make_env，再调用 make_env(config)。
```

`task_id=None` 保留原来的两物体 smoke 环境；五任务需要显式 task ID。
`env.task_description` 提供对应语言指令，`env.task_name` 是完整 suite/name。
任务在一个 batch 中保持一致，不在 reset 时自动切换任务。

参考源码：LW-BenchHub `b2bcb2d00edef691f9fcc49039cbf0bcc7464605`，
`lw_benchhub_tasks/lightwheel_libero_tasks/{libero_spatial,libero_10}` 中的同名任务。
原生实现不导入它们；模拟器只通过任务定义读取本地 USD。

## 2026-09-22 CPU

执行代理在远端 `/opt/venvs/carrot` Python 3.12 下运行旧接口测试与新增任务目录/谓词测试：
16 passed（1.99s）。GPU 任务检查尚未运行，不沿用此前单任务 PASS。

## 资产准备

已下载对象缓存：Bowl008、Plate012、Bowl009、Cookies002、Bottle054、Pot086。
下载使用 `lightwheel_sdk.loader.object_loader.acquire_by_registry`，类型为 `objects` / `USD`；
仅准备资产时使用 SDK，运行环境不导入 LW-BenchHub。

缓存就绪后，用以下脚本创建一个新的资产目录；不会覆盖现有输出或修改缓存：

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
