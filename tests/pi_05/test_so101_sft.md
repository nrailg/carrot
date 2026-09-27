# SO101 SFT 数据与推理契约

## 目的与关键断言

使用模拟数据、tokenizer 和小模型验证四项契约，不加载真实 checkpoint 或连接机械臂。

- 数据工厂保留 revision、30 FPS 动作窗口、双相机字段及 padding mask。
- 训练和推理共用图像、状态、任务 token 变换；第三视角无效，padding 不参与 loss。
- 六维绝对位置动作正确归一化和还原，不误作相对状态的增量。
- 配置指定数据集统计量时不误读基座的 14 维统计量；导出布局可装配六关节 policy。

## 运行

只需提供个人 DFS 根目录；公共 runner 激活 `/opt/venvs/carrot`，
从 `${MY_DFS}/work/carrot` 加载源码并设置 `PYTHONPATH`。测试无需 GPU 或下载资源。

```bash
export MY_DFS=/absolute/path/to/personal/dfs
bash "${MY_DFS}/work/carrot/tests/pi_05/test_so101_sft.sh"
```

## 2026-09-27

补齐同名 runner 和档案；Shell 语法及 diff 检查通过。


## 2026-09-27：Gemini 验证

状态：**PASS**；`4 passed in 6.87s`，runner exit=0。

```bash
export MY_DFS=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu
cd "${MY_DFS}/work/carrot"
bash tests/pi_05/test_so101_sft.sh
```

证据：`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/test-runs/so101-20260927T/test_so101_sft.log`。

本轮镜像、源码和环境记录见 [共同环境](../so101_real/test_so101_runtime.md#本轮共同环境与源码)。
