# 完整teacher轨迹的reward与critic优势分解

状态：PASS（诊断计算及线性分解），2026-10-24；实际RL信号尚未重新验收。

使用HZ持久teacher前两条完整观测、SFT500的真实FP32 critic（SFT阶段未训练的初始value head），以及Pro teacher-replay-01中
每控制步IoU/抓取/归还/成功的奖励变化。只传递数KB奖励元数据，teacher图像保持HZ本地复用。
按phase边界与最多5步分段，保留原gamma=.99/lambda=.95的每控制步口径。

分别计算A(reward,V)、A(0,V)、A(reward,0)，断言线性分解；统计原始/标准化优势的正负零、
尺度和首100控制步的奖励贡献。正负号可能仅来自critic误差，不等于任务有学习信号。
这是teacher上的离线诊断，不是旧RL实际在线advantage，不能冒充旧32环境的分段结果。

确认MY_DFS/POLICY_GPU/POLICY_CONFIG/POLICY_CHECKPOINT/TEST_OUTPUT，运行同名.sh并传
`--signals-json`（两条轨迹的seed/controls/success/events）。输出result.json、values.npy、
reward-events.json及完成标记；不修改模型，不启动RL，不复制teacher或覆盖旧产物。

Carrot ba45957加本地未提交诊断；Docker tag未记录，/opt/venvs/carrot、torch2.11.0+cu128。

运行task7739aeea-0114 exit0/262s，H20 GPU2；输出HZ tests/carrot/multicolor_transfer/reward-signal-20261024-01。
1056/1068控制步展开213/215chunks，总reward1.99921/2.03395，非零reward126/142步。
初始动作的纯reward优势8.685e-15/8.528e-15；前100controls最大值2.949e-12/2.896e-12。
该窗口使用实际初始critic得到的优势std=.01283/.01166，正15%/负85%，与去掉全部reward后的
critic-only优势在FP32下相同。完整轨迹原始优势正41.78%/42.79%，标准化后正22.54%/24.19%。
因此正负号不能证明早期动作获得任务反馈；短credit trace将远处抓笔/成功信号严重衰减。
这是SFT初始未训练critic在teacher上的诊断，不是旧RL四轮后的在线分布。输出保留全部values与reward事件。

2026-10-24增加CPU离线credit尺度对照（待运行）：`--credit-input <已有reward-signal输出> --output <新目录>`，
复用奖励与value，按逐控制步比较gamma/lambda=.99/.95、.99/1、1/1。最后一项须等于MC总回报；
不改变正式配置、不加载模型、不启动SFT/RL，只判断长轨迹的信号衰减来自何种折扣。

2026-10-24 CPU离线credit尺度核验PASS：输出HZ tests/carrot/multicolor_transfer/credit-scale-20261024-01/result.json。
逐控制步初始reward-only优势，seeds15/12：原gamma=.99/lambda=.95为8.404e-15/8.260e-15；
仅lambda=1为.00313239/.00306365；gamma=lambda=1为1.99921048/2.03394985，等于未折扣总奖励，断言通过。
lambda逐控制步衰减是早期信号消失的主要来源之一；这只是离线反事实，不表示lambda=1已获在线验收。
没有新GPU任务、模型推理或配置变更；零奖励失败轨迹也不能靠增大lambda获得任务奖励。
