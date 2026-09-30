# SO101 单腕、15FPS与度数部署配置

## 目的与关键断言

- 单腕观测只包含wrist视角，不复制或伪造base视角，保留episode边界/padding。
- 本地无Hub revision的so_follower数据按15FPS构造动作窗口；非法metadata提前失败。
- 15FPS执行节拍扣除发送耗时，机器人构造器显式透传use_degrees与相机配置。
- 动作限位复用真实LeRobot总线的标定换算，度数与归一化模式分别校验raw目标范围；
  夹爪保持百分比，实际关节已越限时不继续下发。
- 使用fake设备和未连接的Feetech总线；不读真实串口、不发送机器人运动命令。

## 运行

```bash
bash "${MY_DFS}/work/carrot/tests/so101_real/test_so101_profiles.sh"
```

同时回归整个`tests/so101_real`，保留旧双相机30FPS归一化配置的行为。

## 2026-09-30

状态：**PASS**。独立 runner：`16 passed in 6.70s`；完整 SO101 CPU 回归：
`53 passed in 7.83s`（profiles 16、deployment 16、runtime 17、client 3、report 1）。

```bash
export MY_DFS=/mnt/ceph-hz1-csp/mm-base-plt2/nrwu
source /opt/venvs/carrot/bin/activate
cd "${MY_DFS}/work/carrot"
export PYTHONPATH="$PWD/src:$PWD/tests:$PWD:${PYTHONPATH:-}"
bash tests/so101_real/test_so101_profiles.sh
python -m pytest -q tests/so101_real
```

- Gemini launcher：`mpi-1784764303-launcher`；Docker image tag 本轮未核实。
- Python 3.12.13、pytest 9.1.1、PyTorch 2.11.0+cu128、LeRobot 0.6.1、
  NumPy 2.3.1、websockets 16.1.1。LeRobot wheel Git commit 未记录。
- OpenPI client 固定源码 commit：`215abfb217dbac7d5f1273282331b9b1866c0479`。
- Carrot HEAD：`005d87293de85e4fa3900c02416dce08bef5e652` 加当前工作区改动；
  通过 Mutagen 同步，运行前对比修改源码/测试的 SHA256。
- 首轮收集缺 pyserial，补齐后两项硬件工厂测试缺 deepdiff；最终离线补齐
  feetech-servo-sdk 1.0.0、pyserial 3.5、deepdiff 8.6.2、orderly-set 5.5.0 后合跑通过。
  来源为 Mac uv 缓存，未升级其他既有依赖；项目 `client` extra 正式声明硬件依赖。
- 证据为上述命令的远程工具输出；未额外保存独立日志文件。
- 本次使用替身与未连接总线，没有真实机器人运动或新 checkpoint 推理。


## 2026-09-30：review清理与模型单位边界回归（准备）

- 范围：显式YAML相机字段、config传递、degree→radian及夹爪百分点→[0,1]，样本/stats同时换算；反向驱动换算、起点容差、导出与握手单位契约。
- 命令：既定Gemini venv中从当前同步源码运行 `python -m pytest -q tests/pi_05/test_so101_sft.py tests/so101_real`。均使用CPU及fake设备，不接触真实串口或相机，不启动训练/模型评估。
- 预期：新单位回归与已有客户端/日志/调度测试通过，旧服务缺单位声明时在硬件连接前拒绝。
- 状态：NOT RUN；源码d16d29a加本轮未提交改动，Docker image tag未记录；实际结果运行后补充。


### 2026-09-30 20:38北京时间：实际结果 PASS

- Gemini task56253f33-0052 exit0：`tests/pi_05/test_so101_sft.py tests/so101_real tests/pi_05/test_pi05_inference.py tests/sft/test_sft_checkpoint.py` 合计92 passed；真实数据同名shell另2 passed，共94项。CPU-only，CUDA_VISIBLE_DEVICES为空；未接触机器人或新模型推理。
- Ruff通过；31个改动源码/配置/shell的Mac与hz1 SHA256一致，shell语法、py_compile和git diff --check通过。实际源码d16d29a加未提交改动，Docker tag未记录。
- 初轮夹爪缩放后固定归一化epsilon造成约1e-5偏差，测试原1e-6零点容差过严；改成有解释的2e-5。随后wrapper子shell因MY_DFS未export失败，已修正执行环境；风格问题已修正。最终测试正常完成，未跳过失败项。
- 完整最终日志及初轮失败日志在 `/mnt/ceph-hz1-csp/mm-base-plt2/nrwu/test-runs/so101_review_cleanup_20260930/`，含final_tests.log和source_hashes.json。
- 新YAML配置/新单位接口准备完成；未启动新训练或真机任务，未commit。旧服务没有单位metadata，后续联调须按当前代码重启。
