# PI0.5 OpenPI parity 测试记录

## 测试思路

- 以官方 OpenPI JAX PI0.5 为基准，对比 Carrot `PI0Policy` 的数值。
- 两个框架使用同一组固定输入和 noise，先只排查一个 diffusion step。
- `.npz` 仅作为服务器上两个独立进程之间的临时数据，不进入仓库。
- 保持 `rtol=1e-3, atol=1e-3`，不通过放宽容差掩盖差异。
- 额外报告 normalized squared L2（Dice distance），用于判断全局相对误差，不替代逐元素断言。
- 对每个 tensor 汇总 absolute error 和相对 OpenPI reference 的 relative error
  `abs(actual - expected) / max(abs(expected), 1e-12)`，报告 P50/P90/P99/Max。

## 测试代码

- `tests/pi05_parity/generate_openpi_jax_golden.py`：在 OpenPI 环境生成单步参考值。
- `tests/test_pi05_openpi_parity.py`：加载临时参考值并与 Carrot 单步结果比较。
  同时以 Markdown 表格报告各中间量的 absolute/relative error 分位数和 Dice distance。
- 被测实现：`src/carrot/models/pi05/model/modeling_pi05.py` 和
  `src/carrot/models/pi05/model/paligemma_with_expert.py`。

## 执行步骤

```bash
cd /mnt/ceph-hz1-csp/mm-base-plt2/nrwu/work/openpi
.venv2/bin/python \
  /mnt/ceph-hz1-csp/mm-base-plt2/nrwu/work/carrot/tests/pi05_parity/generate_openpi_jax_golden.py \
  --checkpoint /mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/Physical-Intelligence/pi05_base \
  --openpi-commit 15a9616a00943ada6c20a0f158e3adb39df2ccac \
  --output /tmp/pi05-openpi-jax-golden.npz

cd /mnt/ceph-hz1-csp/mm-base-plt2/nrwu/work/carrot
CARROT_PI05_OPENPI_GOLDEN=/tmp/pi05-openpi-jax-golden.npz \
CARROT_PI05_OPEN_GIGA_CHECKPOINT=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/open-gigaai/pi05_base \
pytest -v -s --timeout=1800 tests/test_pi05_openpi_parity.py
```

## 结果

- 当前状态：`FAIL`，测试可稳定复现，非 OOM、CUDA、shape 或非有限值错误。
- 首次失败：`image_embeddings` 的既定 `rtol=1e-3, atol=1e-3` 逐元素断言。

| tensor | abs P50 | abs P90 | abs P99 | abs Max | rel P50 | rel P90 | rel P99 | rel Max | Dice distance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| weight_language_embedding_rows | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| weight_action_in_kernel | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| weight_action_in_bias | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| weight_image_patch_kernel | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| weight_image_patch_bias | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| weight_image_position_embedding | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| weight_image_head_kernel | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| weight_image_head_bias | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| image_embeddings | 0.0126953125 | 0.0390625 | 0.09375 | 2 | 0.0216216216 | 0.153846154 | 1.56 | 6.49414062e+10 | 3.89943542e-05 |
| image_embeddings_camera_0 | 0.013671875 | 0.0390625 | 0.09375 | 2 | 0.0218579235 | 0.155172414 | 1.5671917 | 6.49414062e+10 | 3.91433345e-05 |
| image_embeddings_camera_1 | 0.01171875 | 0.0390625 | 0.09375 | 2 | 0.0213675214 | 0.15 | 1.52124137 | 6.07910156e+10 | 3.65975341e-05 |
| image_embeddings_camera_2 | 0.0126953125 | 0.0390625 | 0.09375 | 2 | 0.0217391304 | 0.156028369 | 1.58684327 | 4.51660156e+10 | 4.12161526e-05 |
| language_embeddings | 0 | 0.03125 | 0.125 | 4 | 0 | 0.00534759358 | 0.00746268657 | 0.00775193798 | 4.53225518e-06 |
| suffix_embeddings | 0 | 0.00390625 | 0.015625 | 0.03125 | 0 | 0.010989011 | 0.101797551 | 357.503704 | 3.4371976e-06 |
| adarms_cond | 0.00013743341 | 0.000350413518 | 0.000595237836 | 0.000905677676 | 0.00896037423 | 0.0570162322 | 0.632157809 | 3.28770344 | 3.64770077e-06 |
| first_v_t | 0.00388103724 | 0.0137346916 | 0.0446174517 | 0.0605726242 | 0.00586205687 | 0.0434710842 | 0.455411742 | 7.75838027 | 5.35283327e-05 |
| one_step | 0.00388103724 | 0.0137346916 | 0.0446174517 | 0.0605726242 | 0.0832973712 | 0.450216938 | 3.07924464 | 107.776151 | 6.06403428e-03 |

- `rel Max` 会被接近零的 OpenPI reference 放大，需结合分位数和 Dice distance 解读。
- 当前证据指向 JAX 与 PyTorch port 的混合精度/舍入路径不同，不支持“抽查权重转换错误”的假设；
  下一步应逐层对齐 dtype 和首个 transformer block 输出。
