# SFT关键动作帧与补齐维度诊断

状态：三轮均完成，前向/梯度契约PASS；闭环根因尚未完整确认。2026-10-24，用户要求先排查实现BUG与advantage，RL仍需完成，SFT迁移评测后置。

在HZ已有teacher的seeds15/12选最初移动、下降、闭爪、转入绘制等18个观测；恢复可信step500
learner.pt中的FP32权重。固定噪声，比较正常ODE和强制25个补齐维度维持训练分布x_t=t*noise
的反事实ODE，记录7D动作误差、末步补齐latent幅度、三个噪声时刻有效/补齐维度的速度MSE。
本实验SFT只监督前7维，而既有Pi05SFTLossFn监督全部模型维度；需实际测量其影响，不能预先
把该差异宣称为闭环零成功根因。反事实只在测试中使用，不修改生产推理或旧模型。

执行：确认MY_DFS、POLICY_GPU、POLICY_CONFIG、POLICY_CHECKPOINT、TEST_OUTPUT，运行同名.sh。
预期输出results.json逐帧证据及completed.json汇总；有限值断言通过不等于闭环成功。
Carrot ba45957加本地未提交诊断，Docker image tag未记录。

第一轮task7739aeea-0102完成/exit0，128秒，HZ tests/carrot/multicolor_transfer/sft-critical-20261024-01。
正常/强制padding三轴MAE分别[.01719,.04863,.05305]/[.01715,.04812,.05325]，并无实质改善；
补齐latent末态绝对均值约.003～.005，不能把未监督补齐维度认作当前零成功根因。
关键移动帧速度MSE显著高于等待帧，offset0在t=.1为.1641，offset40为.001304。

第二轮增加--fit-steps 20：同样18个真实观测固定小样本拟合，保持micro8/GBS256/lr1e-5，
仅20个优化步、不写入旧checkpoint、不用于正式评测；用相同噪声验证动作误差能否下降，
记录sft-fit.jsonl及fit-result.json，验证训练梯度和推理读权重链路。输出sft-critical-20261024-02。

第二轮task7739aeea-0105 exit0/370s：20步loss末值.00844889、grad_norm .14634627；
相同噪声的xyz动作MAE从[.01719,.04863,.05305]到[.01443,.00967,.03978]。梯度能改变推理输出，
但关键帧未完全拟合，不能据此排除所有训练BUG或声称闭环修复。

第三轮新增同输入/噪声的原模型联合attention前向与生产KV-cache前向对照，比较速度和三组
动作参数梯度方向。禁止图像增强干扰这项比较；不更新参数。输出sft-critical-20261024-03。

第三轮task7739aeea-0112 PASS/exit0，191s，H20 GPU3。速度max差.015625、相对L2=.00420178；
梯度cosine：action_in .999385、action_out .999807、expert首层query .999351。
对应梯度相对L2=.0360/.0206/.0382；均为BF16混合精度路径，不能声称逐位相同。
证据forward-contract.json及results.json，SHA256 2b5a9c12c4614924cf08470297827c9781ea539e2325960d9845e8ca7c26aa76。
