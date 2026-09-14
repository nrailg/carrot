# PI0.5 OpenPI parity 测试记录

## 测试思路

- 目标：以官方 OpenPI JAX PI0.5 为基准，验证 Carrot `PI0Policy` port 的推理数值。
- 覆盖的行为：官方 Orbax checkpoint 与转换后的 Carrot checkpoint，在相同预处理后图像、
  token、mask、state 和固定 Gaussian noise 下的一步去噪及十步 Euler sampling。
- 不覆盖的边界：tokenizer、图像 resize、normalization、action unnormalization 和 LeRobot 实现。
- 隔离方式：OpenPI JAX 与 Carrot PyTorch 分进程执行；JAX 先生成包含输入、输出和版本元数据的
  `.npz` golden，pytest 再读取，避免两个框架争用 GPU 或各自生成不同随机数。
- 预期通过条件：输出 shape 相同且值有限；一步结果满足 `rtol=1e-3, atol=1e-3`，十步结果满足
  `rtol=1e-2, atol=5e-3`。
- 预期失败条件：golden 元数据或 shape 不符合约定，出现非有限值，或任一数值断言超出容差。

## 测试代码

- 新增 golden 生成器：`tests/pi05_parity/generate_openpi_jax_golden.py`。
  - 使用 `Pi0Config(pi05=True)` 和官方 `restore_params(..., dtype=jnp.bfloat16)` 加载 Orbax 参数。
  - 由 NumPy 固定种子生成唯一一份输入和 noise，并保存一步、十步 JAX 输出。
- 新增测试：`tests/test_pi05_openpi_parity.py`。
  - 选择表达式：`tests/test_pi05_openpi_parity.py`。
  - 直接通过 `PI0Policy.from_pretrained` 加载转换后的 checkpoint，将 golden 中 NHWC 图像转换成
    Carrot 使用的 NCHW，再比较一步和十步输出。
- 被测代码：`src/carrot/models/pi05/model/modeling_pi05.py`、
  `src/carrot/models/pi05/model/paligemma_with_expert.py`。
- 相关基线提交：`bd0891c18643c6c1682f1ce74d54f3c941180e80`。

## 执行步骤

1. 代码版本与同步检查：确认本地工作树；通过现有 lsyncd 将 `/home/nrwu/work/carrot` 同步到
   `/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/work/carrot`，运行前比较测试文件 hash。
2. OpenPI 环境与 checkpoint：
   - OpenPI repo：`/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/work/openpi`，同步自干净的官方仓库
     commit `15a9616a00943ada6c20a0f158e3adb39df2ccac`。
   - 官方 checkpoint：
     `/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/Physical-Intelligence/pi05_base`。
   - 转换 checkpoint：
     `/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/open-gigaai/pi05_base`。
3. 生成官方 JAX golden：

   ```bash
   cd /mnt/ceph-hz1-csp/mm-base-plt2/nrwu/work/openpi
   uv run python \
     /mnt/ceph-hz1-csp/mm-base-plt2/nrwu/work/carrot/tests/pi05_parity/generate_openpi_jax_golden.py \
     --checkpoint /mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/Physical-Intelligence/pi05_base \
     --openpi-commit 15a9616a00943ada6c20a0f158e3adb39df2ccac \
     --output /mnt/ceph-hz1-csp/mm-base-plt2/nrwu/work/carrot/docs/test-manifest/pi05-openpi-parity-20260914.artifacts/openpi-jax-golden.npz
   ```

4. 在 Carrot 环境执行正式 pytest：

   ```bash
   cd /mnt/ceph-hz1-csp/mm-base-plt2/nrwu/work/carrot
   export PYTHONPATH="$PWD/src:$PWD/tests:$PYTHONPATH"
   export CARROT_PI05_OPENPI_GOLDEN="$PWD/docs/test-manifest/pi05-openpi-parity-20260914.artifacts/openpi-jax-golden.npz"
   export CARROT_PI05_OPEN_GIGA_CHECKPOINT=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/hf-hub/open-gigaai/pi05_base
   set -o pipefail
   pytest -v -s --timeout=1800 \
     --junitxml=docs/test-manifest/pi05-openpi-parity-20260914.artifacts/junit.xml \
     tests/test_pi05_openpi_parity.py \
     2>&1 | tee docs/test-manifest/pi05-openpi-parity-20260914.artifacts/pytest.log
   pytest_status="${PIPESTATUS[0]}"
   printf '%s\n' "$pytest_status" \
     > docs/test-manifest/pi05-openpi-parity-20260914.artifacts/pytest.exit-code
   exit "$pytest_status"
   ```

5. Ray：此模型级单 GPU 测试不依赖 Carrot distributed/Ray，不启动 Ray。

## 实际结果

- 状态：`BLOCKED`。
- 每次运行：
  - 本地预检：2026-09-14，`pytest -q tests/test_pi05_openpi_parity.py`，退出码 `2`；
    collection 时因本机 Python 3.11 环境没有 `torch` 而停止，未执行测试逻辑。
  - Gemini：2026-09-14，由 Luna xHigh 尝试连接；`remote_status` 无 active session，
    `remote_connect(container="mpi-launcher@mpi-1742693300-launcher")` 在执行远端命令前失败，
    因此无远端进程退出码或 pytest 汇总。
- 失败、阻塞或偏差及判断：本地环境不是 Carrot SFT/GPU 测试环境，不能用于数值验证；
  该 collection failure 不代表测试代码或模型失败。Gemini 端未提供 token，且本地未配置
  `GEMINI_PROJECT_ID`、`GEMINI_PROJECT_KEY`、`GEMINI_USER`，错误为
  `No token provided and auto-generation not configured`。golden 和 pytest 均未启动。
- 证据：本地预检的 `local-collection.log`、`local-collection-junit.xml` 和
  `local-collection.exit-code` 已保存到
  `docs/test-manifest/pi05-openpi-parity-20260914.artifacts/`；Gemini 在连接前失败，无远端日志。
- 清理动作与残留：未连接远端，未暂停 dguard、未占用 GPU、未启动 Ray，无远端残留。

## 后续

- 若一步即失败，优先比较 image embedding、prefix KV cache、action/time embedding 和首个
  denoise `v_t`，不先放宽容差。
- Carrot 对 OpenPI JAX 对齐后，再把 LeRobot 作为第三条实现加入诊断矩阵。
- 提供有效的 `./gemini-go <container> <token>`，或配置 `GEMINI_PROJECT_ID`、
  `GEMINI_PROJECT_KEY`、`GEMINI_USER` 后，按“执行步骤”中的原命令继续。
