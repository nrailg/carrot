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
