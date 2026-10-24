# 随机五条旧示范的 Isaac 回放

用户要求直接观看现有示范，检查所谓停顿。只回放已记录动作，不调用新teacher生成动作，不修改原数据或成功条件。

2026-10-24：从全部64条旧成功示范均匀无放回抽5条，随机seed为569328295，抽样发生在查看停顿统计之前。旧数据包含25条单色、21条双色、18条三色。

| 视频 | episode | 环境seed | 颜色数 | 控制步数 | 首个末端近静止区间 |
| --- | --- | --- | --- | --- | --- |
| 1 | 0039 | 1049 | 2 | 2245 | 1.70–12.50秒 |
| 2 | 0021 | 1033 | 1 | 1111 | 1.85–12.70秒 |
| 3 | 0024 | 1035 | 1 | 1131 | 1.85–12.70秒 |
| 4 | 0053 | 1064 | 3 | 3238 | 1.85–12.70秒 |
| 5 | 0004 | 17 | 1 | 1065 | 1.70–12.50秒 |

PRO原始路径：个人DFS下`experiments/carrot/pi05_multicolor_transfer/pilot-20261023-02/renderer/demos`；五条actions.npy/state.npy的SHA与H20的`pilot-20261023-02-recovery-01/demos`逐一匹配，未重新复制40GB原数据。

视频保持每控制步一帧、20fps正常时间尺度，未剪停顿或加速。每帧显示时间、源phase、平移命令范数、当前夹爪间距及夹爪命令，左主相机/右腕相机。RECORDED为原数据相机帧，ISAAC REPLAY为重新执行原动作得到的帧。黄色LOW XYZ COMMAND表示三维平移命令范数小于0.01；夹爪仍可能运动。区间统计额外要求末端每步平移小于0.2mm并持续至少1秒，不等于整个机器人所有关节静止。

脚本：replay_demos.py。先`prepare --root <旧demos> --output <新输出>`导出抽样清单和原相机视频；再对index0–4运行`replay --output <同一输出> --index <索引> --source <冻结源码> --robot-usd <本地USD>`。每次激活carrot venv、设置PYTHONPATH与空闲PRO GPU。

本轮输出：PRO个人DFS下`experiments/carrot/pi05_libero_draw_sft_transfer/demo-replay-20261024-01`。回放脚本冻结在`sources/replay-20261024-01/replay_demos.py`，SHA c2e07bd013761a9beb44bd4b6fb11b1cff660ec2dbbaf73e5061e577edcbbfa8；场景来自`sources/sft-mix-20261024-01`。基线6be3d5a加已记录修改，镜像tag未记录。

原录像导出任务6b56755c-0383完成，48秒/exit0。Isaac回放1/2任务0385/0386分别282秒/154秒，exit0，均成功且drops=0。初次3/4/5实例启动未推进，已停止任务0387/0388/0389并保留启动日志；未替换抽中的数据。第4条改在GPU3由0393执行（346秒/exit0），第3/5条在GPU2由0395串行执行（273秒/exit0）。五条回放均产出视频；1/2/3/5成功，drops=0；第4条在3072controls即153.6秒掉笔终止，success=false，原示范3238controls记录为成功。没有替换该样本，也未把进程exit0当作物理成功。原始第4条录像一并提供；回放差异原因未定位。用户GPU0和原GPU1任务未停止。

本地观看文件：`/Users/wujunyu/.codex/visualizations/2026/10/24/demonstration-replay/`。

停顿存在、示范可物理回放、策略能否学会是不同问题；此回放不能单独证明停顿导致学习失败，也不把它作为启动训练的前置条件。

五条本地视频SHA与远端一致；ffprobe核验全部20fps，帧数与实际执行控制步一致。汇总replay-summary.json，任务日志task-0383/0385/0386/0393/0395.log均持久化。
