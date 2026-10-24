# 全参数 SFT GPU 预检

在正式训练前，从共同base临时执行两个GBS256的更新，检查视觉编码器、VLM和action expert的梯度有限且非零，代表参数确实更新。临时权重丢弃，不算正式训练路线。

执行：设置TRAIN_CONFIG、TEST_OUTPUT、MY_DFS和8张空闲GPU后，`bash tests/libero_draw_transfer/check_model.sh`。不改变正式GBS/micro/DP。

预期：两个步骤均通过，result.json的passed=true。任何rank失败不得启动正式训练。

状态：NOT RUN；Gemini连接信息待提供。

Docker image与实际运行commit：未记录；本地基线6be3d5a加实验改动。

## 2026-10-24 H20 8卡实测

真实LIBERO数据、GBS256/micro4/GAS8、完整SFT参数更新：PASS。任务9f2afe6d-0125退出0，121秒。两个步骤rank0 loss为0.0739441/0.0895386，总梯度范数0.735212/0.578681；视觉/VLM/expert各组都有有限非零梯度，代表参数最大更新约1e-5。临时权重未保存。

证据：H20个人DFS `/mnt/ceph-hz1-csp/mm-base-plt2/nrwu` 下 `experiments/carrot/pi05_libero_draw_sft_transfer/sft-mix-20261024-01/h20-preflight/gradient-check/result.json`，任务输出归档为同级上层 `task.log`。固定101文件源码位于 `experiments/carrot/pi05_libero_draw_sft_transfer/sources/sft-mix-20261024-01`，source-manifest.json记录SHA。

环境：torch2.11.0+cu128、transformers5.5.4、lerobot0.6.1、ray2.58.0、safetensors0.8.0；实际Docker tag与上游git commit未记录。H20 head本地dguard已在检查结束后恢复。此结果验证全参数梯度；混合新版示范的实际路径还需在数据就绪后执行。
