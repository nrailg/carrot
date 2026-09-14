# PI0.5 OpenPI parity 测试记录

## 测试思路

- 以官方 OpenPI JAX PI0.5 为基准，对比 Carrot `PI0Policy` 的数值。
- 加入 OpenPI 当前 main（`215abfb217dbac7d5f1273282331b9b1866c0479`）的官方 PyTorch
  实现，形成 JAX、官方 PyTorch、Carrot 三路对比。
- 三个实现使用同一组固定输入和 noise，先只排查一个 diffusion step。
- `.npz` 仅作为服务器上独立进程之间的临时数据，保留在个人 Ceph 临时目录，不进入仓库。
- 保持 `rtol=1e-3, atol=1e-3`，不通过放宽容差掩盖差异。
- 额外报告 normalized squared L2（Dice distance），用于判断全局相对误差，不替代逐元素断言。
- 对每个 tensor 汇总 absolute error 和相对 OpenPI reference 的 relative error
  `abs(actual - expected) / max(abs(expected), 1e-12)`，报告 P50/P90/P99/Max。
- 分别报告三组差异：Carrot 对 JAX、官方 PyTorch 对 JAX、Carrot 对官方 PyTorch；
  从 image embedding、suffix embedding、AdaRMS、首个 `v_t` 到一步采样定位首个分叉点。
- 官方 PyTorch 使用上游转换器生成的临时 checkpoint，并按上游加载流程恢复混合参数 dtype；
  转换结果和两份 `.npz` 都只保存在服务器个人临时目录。

## 测试代码

- `tests/pi05_parity/generate_openpi_jax_golden.py`：在 OpenPI 环境生成单步参考值。
- `tests/pi05_parity/generate_openpi_pytorch_golden.py`：在 OpenPI 当前 main 的 PyTorch
  环境加载官方转换 checkpoint，复用 JAX 临时文件中的输入并生成单步参考值。
- `tests/test_pi05_openpi_parity.py`：加载临时参考值并与 Carrot 单步结果比较。
  同时以 Markdown 表格报告三路各中间量的 absolute/relative error 分位数和 Dice distance，
  保留严格逐元素断言。
- 被测实现：`src/carrot/models/pi05/model/modeling_pi05.py` 和
  `src/carrot/models/pi05/model/paligemma_with_expert.py`。

## 执行步骤

```bash
bash /root/dguard/dguard.sh stop 120

O=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/pi05-openpi-current
C=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/work/carrot
V=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/pi05-parity-venv
T=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/pi05-parity-temp
OLD_SITE=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/work/openpi/.venv2/lib/python3.12/site-packages

PYTHONPATH="$O/src:$O/packages/openpi-client/src:$C/src:$C/tests:$OLD_SITE" \
"$V/bin/python" "$C/tests/pi05_parity/generate_openpi_jax_golden.py" \
  --checkpoint /mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/Physical-Intelligence/pi05_base \
  --openpi-commit 215abfb217dbac7d5f1273282331b9b1866c0479 \
  --output "$T/pi05-openpi-jax-golden.npz"

PYTHONPATH="$O/src:$OLD_SITE" \
"$V/bin/python" "$O/examples/convert_jax_model_to_pytorch.py" \
  --checkpoint-dir /mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/Physical-Intelligence/pi05_base \
  --config-name pi05_aloha \
  --output-path "$T/pi05-openpi-pytorch-checkpoint" \
  --precision bfloat16

PYTHONPATH="$O/src:$O/packages/openpi-client/src:$C/src:$C/tests:$OLD_SITE" \
"$V/bin/python" "$C/tests/pi05_parity/generate_openpi_pytorch_golden.py" \
  --checkpoint "$T/pi05-openpi-pytorch-checkpoint" \
  --jax-golden "$T/pi05-openpi-jax-golden.npz" \
  --openpi-commit 215abfb217dbac7d5f1273282331b9b1866c0479 \
  --output "$T/pi05-openpi-pytorch-golden.npz"

cd "$C"
PYTHONPATH="$C/src:$C/tests" \
CARROT_PI05_OPENPI_GOLDEN="$T/pi05-openpi-jax-golden.npz" \
CARROT_PI05_OPENPI_PYTORCH_GOLDEN="$T/pi05-openpi-pytorch-golden.npz" \
CARROT_PI05_OPEN_GIGA_CHECKPOINT=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/open-gigaai/pi05_base \
"$V/bin/python" -m pytest -v -s --timeout=1800 tests/test_pi05_openpi_parity.py
```

## 结果

### OpenPI current-main 三路对比

- `FAIL`：三路诊断完整执行；首个严格断言失败在
  `openpi_pytorch_vs_jax/image_embeddings`，不属于 OOM、CUDA、shape 或非有限值错误。
- 官方源码为独立 checkout，`HEAD=215abfb217dbac7d5f1273282331b9b1866c0479`；
  converter 和 golden 生成器均确认从该目录 import `openpi`。
- 官方 PyTorch 转换后再按上游加载策略恢复 dtype：image patch/position 与 action projection
  为 FP32，image projector 与 language embedding 为 BF16。
- 官方 PyTorch 对 JAX 的八组抽查权重 FP32 数值完全一致。Carrot 的对应 FP32 权重与二者
  Dice 为 `9.41e-07` 至 `1.48e-06`，转成 BF16 后逐位一致；说明 Carrot checkpoint
  保留了 BF16 格点之间的额外 FP32 尾数。

| comparison/tensor | abs P50 | abs P90 | abs P99 | abs Max | rel P50 | rel P90 | rel P99 | rel Max | Dice distance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| carrot_vs_jax/image_embeddings | 0.0129234977 | 0.0398657024 | 0.0938975656 | 2.12535095 | 0.0215880906 | 0.153628545 | 1.56009118 | 6.51838109e+10 | 3.79534734e-05 |
| openpi_pytorch_vs_jax/image_embeddings | 0.01171875 | 0.0390625 | 0.09375 | 2 | 0.0208333333 | 0.147540984 | 1.5105911 | 4.4921875e+10 | 3.61103573e-05 |
| carrot_vs_openpi_pytorch/image_embeddings | 0.0115200877 | 0.0350503653 | 0.0802722502 | 1.44098663 | 0.0192396867 | 0.13580443 | 1.38297537 | 33622.7405 | 2.83228836e-05 |
| carrot_vs_jax/language_embeddings | 0.00806438923 | 0.0340155602 | 0.0830569077 | 2.72363281 | 0.00168167422 | 0.00386154292 | 0.00530503543 | 0.00640764186 | 3.00196131e-06 |
| openpi_pytorch_vs_jax/language_embeddings | 0 | 0 | 0.03125 | 0.25 | 0 | 0 | 0.00529100529 | 0.00625 | 2.037637e-07 |
| carrot_vs_openpi_pytorch/language_embeddings | 0.00804376602 | 0.0337877274 | 0.081192131 | 2.72363281 | 0.00167406501 | 0.00384351241 | 0.00529130746 | 0.00632115414 | 2.93963553e-06 |
| carrot_vs_jax/suffix_embeddings | 0.0012422055 | 0.00317515135 | 0.00520447213 | 0.0113324672 | 0.0016653212 | 0.010149103 | 0.102998826 | 356.791667 | 1.41553379e-06 |
| openpi_pytorch_vs_jax/suffix_embeddings | 0.000359535217 | 0.000939136744 | 0.00154471636 | 0.0027012825 | 0.000487417259 | 0.00230074613 | 0.0225003857 | 30.8203125 | 1.22551721e-07 |
| carrot_vs_openpi_pytorch/suffix_embeddings | 0.0012164712 | 0.00309302807 | 0.00513483524 | 0.0101762563 | 0.00163055792 | 0.0100815109 | 0.0999952581 | 47.4644776 | 1.36107921e-06 |
| carrot_vs_jax/adarms_cond | 0.00013743341 | 0.000350413518 | 0.000595237836 | 0.000905677676 | 0.00896037423 | 0.0570162322 | 0.632157809 | 3.28770344 | 3.64770077e-06 |
| openpi_pytorch_vs_jax/adarms_cond | 5.44358045e-07 | 1.42995268e-06 | 2.24407762e-06 | 5.84125519e-06 | 3.40405562e-05 | 0.00021606385 | 0.0020641522 | 0.0219926155 | 5.80195891e-11 |
| carrot_vs_openpi_pytorch/adarms_cond | 0.000137939292 | 0.000349964853 | 0.000595445007 | 0.00090637058 | 0.00893911173 | 0.0568874754 | 0.632482153 | 3.25464508 | 3.64747644e-06 |
| carrot_vs_jax/first_v_t | 0.00388103724 | 0.0137346916 | 0.0446174517 | 0.0605726242 | 0.00586205687 | 0.0434710842 | 0.455411742 | 7.75838027 | 5.35283327e-05 |
| openpi_pytorch_vs_jax/first_v_t | 0.00333453715 | 0.00906790979 | 0.0156058836 | 0.0251698494 | 0.00490048645 | 0.0249642625 | 0.285944359 | 3.16669695 | 1.36485342e-05 |
| carrot_vs_openpi_pytorch/first_v_t | 0.00362247229 | 0.0137272969 | 0.037278648 | 0.0443056226 | 0.00526516198 | 0.0419252675 | 0.484716442 | 5.72519076 | 4.44139097e-05 |
| carrot_vs_jax/one_step | 0.00388103724 | 0.0137346916 | 0.0446174517 | 0.0605726242 | 0.0832973712 | 0.450216938 | 3.07924464 | 107.776151 | 0.00606403428 |
| openpi_pytorch_vs_jax/one_step | 0.00333453715 | 0.00906790979 | 0.0156058836 | 0.0251698494 | 0.0705956231 | 0.356180834 | 2.20201406 | 127.013598 | 0.00147571711 |
| carrot_vs_openpi_pytorch/one_step | 0.00362247229 | 0.0137272969 | 0.037278648 | 0.0443056226 | 0.0772736456 | 0.421038152 | 3.9895413 | 92.0454943 | 0.0049684793 |

- 三个 camera 的 image embedding Dice 范围：
  - Carrot 对 JAX：`3.58e-05` 至 `4.01e-05`；
  - 官方 PyTorch 对 JAX：`3.45e-05` 至 `3.75e-05`；
  - Carrot 对官方 PyTorch：`2.72e-05` 至 `2.93e-05`。
- 官方 PyTorch 本身也从 image embedding 开始偏离 JAX，但其一步输出 Dice
  `1.47571711e-03`，比 Carrot 对 JAX 的 `6.06403428e-03` 更小。
- Carrot 与官方 PyTorch 在 image embedding 上彼此更近，但 AdaRMS、首个 `v_t` 和一步输出
  仍有额外差异。下一步需先用 BF16-roundtrip 后的同值权重重跑 Carrot，隔离 checkpoint
  FP32 尾数与实现路径两种因素。

### 初始 JAX 对 Carrot 对比

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
