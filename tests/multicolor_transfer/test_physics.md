# 真实多色笔接触与抓换笔

状态：PASS，2026-10-23，第八轮完成两个环境的真实两色抓画归位。

hold 50 步要求无笔迹；scripted 通过真实 7D EEF/gripper 接口，在两个并行环境完成两色抓取、
接触绘制、放回和松手，IoU≥.72、drops=0。partial reset 检查另一环境的 robot/pen/tip 位姿、画布
和控制计数不变；base/wrist 图像非黑。`rollout.mp4` 与 `trace.jsonl` 用于复核接触和轨迹。

运行前检查 Pro5000 GPU 进程并暂停本轮 dguard，保留其他人的仿真进程。设定 `MY_DFS`、
`TEST_OUTPUT`、`SIMULATOR_GPU` 后执行 `bash tests/multicolor_transfer/test_physics.sh`。
三色示范课程仍需后续正式采集验收；两色预检通过不等于 64 个示范齐备。

Carrot 基线 `ff9e882` 加当前未提交修改；Isaac Sim 依赖目标 6.0.1.0。
实际 Docker image tag、源码 SHA、运行时间、MCP task ID 和结果路径：待运行后记录。

首轮 `80e65273-0199`：hold 与 partial reset 通过；scripted 在 rack 接近点停滞，
eef 残差约 4mm 超过原先 3mm 的 waypoint 阈值，未触发闭爪。
已中断此轮，保留 `.../tests/carrot/multicolor_transfer/physics-20261023-01/trace.jsonl` 和视频。
抓放 waypoint 容差改为 6mm，绘制仍为 3mm；进程内显式选择单一 NVIDIA Vulkan ICD，
避免容器两个 ICD 文件造成同一 GPU 重复枚举。准备使用全新目录重跑。

第二轮 `80e65273-0207`（physics-20261023-02）：真实抓取成功，接触出墨后判为掉笔。
根因是持续夹持误用了离架高度阈值；已分开 lift 与 grasp 判定。
第三轮 `80e65273-0212`（physics-20261023-03）：掉笔误判消失，第一色画完后IoU仅0.416/0.614，
未达0.72，因此未进入归位。中断并保留轨迹；未降低验收阈值。
第四轮 `80e65273-0219`（physics-20261023-04）中断：重复补偿导致上漂。
Workbench 基类已禁用机械臂重力，额外施加 generalized gravity force 是错误的；已移除并加初始化断言。
第五轮：保留原关节控制，示范策略按实际笔尖闭环跟踪参考，避免夹持变形造成路径偏移。
动作仍经过同一7D控制接口，未直接修改任何笔/机械臂位姿；渲染仍为Pro5000。

第五轮 `80e65273-0226`（physics-20261023-05）未通过：笔尖跟踪改善，但夹持中笔杆倾斜，
抬笔回正时沿画布滑动产生额外划痕，第一色最终IoU约0.64/0.68。保留canvas/trace证据后中断。
第六轮：笔尖独立摩擦材质static0.12/dynamic0.1/min combine；手指PD为1200/40，
机械臂PD仍800/60。没有改动抓放或IoU验收条件。

第六轮 `80e65273-0231`：第一色IoU达到0.8090/0.8025，但归位松爪后笔倾倒，退出1。
第七轮 `80e65273-0239`：笔架内宽收至22mm、壁高55mm，归位按笔杆中心修正位置并稳定松手。
seed8完整两色成功：IoU0.7958025、2 picks/2 returns、零掉笔、1963控制步。
seed7两色IoU0.8057883，但第二支笔横向搬运后倾斜，插架时被碰倒；1791步失败，退出1。
两轮输出均位于Pro5000个人DFS的tests/carrot/multicolor_transfer/physics-20261023-06及-07。

第八轮 `80e65273-0248`（physics-20261023-08）：示范归位减速，通过真实旋转动作纠正笔轴倾角，
按实际夹持偏移调整插入位置；松手期间保持末端位姿。未改变笔位姿、重力、接触或成功阈值。
同时按用户要求将画稿仅放在场景参考板上，移动参考板/主相机以便224px图像辨认；不再额外输入参考图。
本轮scene SHA-256 d03726218a9019ffcba4b05b6cbe1846f454fc1534ac31514cb254d04d25607e。
第八轮exit0：seed7/8均成功，IoU分别0.803491/0.795135，均2 picks/2 returns、drops=0；
控制步分别2155/2161，contact_samples为4263/4345。hold、partial reset、双相机图像检查均通过。
证据为该目录result.json、run.log、trace.jsonl、rollout.mp4和最终canvas/base/wrist图像。
