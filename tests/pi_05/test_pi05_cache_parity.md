# PI0.5 full forward 与条件 KV cache 对照

## 2026-10-03：排查准备

目标：不追加训练，比较相同权重、prefix、suffix、mask、position 和精度下的
joint full forward 与 prefix prefill + action forward。优先排除 audit 的 backend 混用。

- 生产 `sample_actions()` 将 prefix/action attention 均设为 eager；旧 audit 直接
  prefill，模型构造默认 prefix SDPA，full forward 固定 eager。
- 新 audit 默认显式 eager，可用 `--prefix-attention sdpa` 单独复现旧设置。
- 同一已训练 frame53/noise1053 checkpoint 对照，不覆盖旧输出；原 FP32 buffers 保留。
- 小模型测试涵盖 FP32/BF16、masked prefix、条件单向读取和缓存不被 suffix 修改。

命令：`MY_DFS=<当前 Gemini 检测值> bash tests/pi_05/test_pi05_cache_parity.sh`。
真实 checkpoint：运行 `recipes/pi05_sft_so101_fit_validation/audit.py`，同一 case 分别
指定 `--prefix-attention eager` / `sdpa`，结果保存在新测试证据目录。

状态：**NOT RUN**。当前代码基线 a9dd5c55f04026be313a55f2215346f02784377c 加未提交诊断修改。
实际 Docker tag、上游版本及运行结果待核验回填。

## 2026-10-03：运行与修复结果

**PASS**。GPU H20，Python3.12.13，torch2.11.0+cu128，transformers5.5.4，pytest9.1.1。
Carrot基线a9dd5c55f04026be313a55f2215346f02784377c加本轮修改；
参考OpenPI commit215abfb217dbac7d5f1273282331b9b1866c0479；Docker tag未记录。

1. 修复前 `test_sdpa_respects_block_mask_with_bf16_queries` FAIL：
   改动禁止读取的后block导致前block输出2048/2048元素变化，max8.8125。
2. 仅对SDPA浮点mask转换为query.dtype后，原block隔离测试及full/cache对照3/3 PASS。
3. 扩展FP32/BF16 × float/bool mask共4项，加full/cache2项；连同既有
   `test_pi05_inference.py`、`test_so101_sft.py`、`test_so101_fit_controls.py`、
   `test_so101_single_frame_recipe.py`共48/48 PASS，17.96s。
4. Ruff最初仅有新增行宽/import格式问题，修正后通过；bash -n与py_compile通过。

证据根：`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/test-runs/so101_cache_diagnosis_20261003/`。
原模型源码保存在`source_before_fix/`，audit运行版本快照及结果均保留：
`eager.json`、`sdpa_before_fix.json`、`sdpa_after_fix.json`。
默认eager同precision缓存与full前五轴MAE均约0.02–0.09°；旧SDPA误差精确复现；
修复SDPA后前五MAE[0.02739,0.06163,0.10193,0.07087,0.01166]°。
SDPA prefix实际内核：BF16用cuDNN，FP32用efficient attention。

`sdpa_mask_probe.json`用显式FP32 QK/softmax/V参考，只比较合法query：
cuDNN+FP32 mask max0.857032，cuDNN+BF16/bool mask max0.003536；math三者均0.003536。
不是generic block mask不受支持，也没有把37.55°误差当成正常舍入误差。

`--rounded-weight-fp32-probe`进一步区分参数舍入和运算精度；
BF16可表示的参数值转回FP32运算，full MAE[0.02116,0.06865,0.09663,0.04767,0.02208]°。
原FP32 master权重full MAE仍[1.44334,1.68998,0.39601,0.36083,0.17860]°。
本轮未自动改变推理默认precision、checkpoint导出格式或已有checkpoint。
与原始Mac Parquet的3次audit/30份rollout独立复算见`independent_validation.json`。

## 2026-10-03 SDPA dtype contract: PASS

- 按用户要求将底层自动转换改为 assert：浮点 mask dtype 与 Q 相等，bool mask 合法。
- audit 的 prefix mask 在调用方显式转换为当前计算 dtype；正常 eager 路径不改。
- FP32/BF16 × float/bool 的四组 block 隔离测试继续运行；两种错误浮点 dtype 组合须在 SDPA 前抛出 AssertionError。
- 源基线 a9dd5c55 加未提交改动，Docker tag 未记录；使用现有 test_pi05_cache_parity.sh。

- 实际执行现有 .sh：6 passed in17.29s，exit0；错误 float dtype 两方向断言和四组合法mask隔离均通过。
- Gemini32434904/H20/torch2.11.0+cu128/transformers5.5.4，Docker tag未记录；源码a9dd5c55加未提交改动，三源码Mac/GPU SHA256一致。Ruff测试文件/三文件compile/diff PASS。
- 证据MY_DFS/test-runs/so101_cache_diagnosis_20261003/dtype_assert/{regressions_6.log,source_hashes.json,三源码快照}。本轮不重跑真实checkpoint audit，不追加训练。
