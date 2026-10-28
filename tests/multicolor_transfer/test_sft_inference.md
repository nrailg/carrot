# SFT示范动作与推理路径诊断

状态：诊断执行完成，2026-10-24；动作变换契约通过，闭环效果尚待独立评测。

用户指出SFT后缺少闭环成功率验收。本诊断先检查已有示范上的动作预测与推理实现，
不训练模型，不改变正式RL，不以离线MAE或契约断言通过声称任务成功。

各选两条一/二/三色成功示范，每个子任务取两个内部观测；读取HZ已有完整teacher，
不跨池传数据。对step50/100/250/500分别测10步ODE、原生10步ODE与RL的5步SDE。
检查动作归一化逆变换误差小于1e-5，预测有限；记录前五步各维MAE/P95、夹爪符号一致率、
越界比例和native/封装ODE差异。相同checkpoint间使用固定样本和随机种子。

启动：确认H20空闲GPU与持久DFS，设置MY_DFS、POLICY_CONFIG、TEST_OUTPUT、POLICY_GPU，
运行bash tests/multicolor_transfer/test_sft_inference.sh。用户要求dguard保持on，按该授权执行。

源码：ba45957加本诊断；Docker image tag未记录。结果路径与实际执行结果待补。

H20 session7739aeea，task7739aeea-0073；72个示范观测，step50/100/250/500全部完成。
输出：/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/tests/carrot/multicolor_transfer/sft-inference-20261024-01。
归一化往返最大误差1.19e-7；step500 ODE三轴动作MAE=[.007858,.005631,.013805]，
SDE为[.018108,.015573,.023245]；夹爪符号均100%。动作单位需乘0.05m才是命令位移误差。
step100 ODE MAE=[.015508,.015495,.024249]，SDE=[.025269,.025099,.034136]。
原生与封装ODE的最大绝对差：step50=.041386，100=.007396，250=.007323，500=.008787。
两路径时间离散/精度细节未做进一步归因，不能写成完全一致；逐帧预测保存在npy，完整统计results.json。
这些离线结果不足以说明训练内闭环成功，不构成SFT通过验收或选定RL初始化的证据。

2026-10-24追加：修正FlowPolicy确定性ODE使用与原生完全相同的累计时间/步长，
新增native-vs-wrapper <1e-5断言；额外记录真实create_libero_policy部署路径与训练侧的同噪声动作差异。
只复测step500，输出sft-inference-20261024-02。此次运行前状态NOT RUN。

sft-inference-20261024-02 task7739aeea-0091 FAIL：仅对齐累计时间仍有.00800598差异，
不能认定时间步是唯一来源；该失败保留。下一轮确定性分支直接共用PI0Pytorch.sample_actions，
避免BF16速度输出在两套Euler实现中发生不同的类型提升。SDE训练分支不改。
复测step500输出sft-inference-20261024-03；执行前NOT RUN。

sft-inference-20261024-03 task7739aeea-0094 PASS/125s，72观测，step500：
native-vs-wrapper最大差0，归一化往返1.19e-7；真实部署与训练侧同噪声前8观测最大差.00527716。
该部署差异仍需与闭环分开判断。SDE三轴MAE未变，确定性分支复用原生ODE后回归通过。
