# Isaac Sim Getting Started Robot

状态：PASS（2026-10-26，真实 GPU 运行与 MP4 录制完成）。

入口直接调用教程 Python；验证正常退出、四轮日志及 400 帧视频完整性。
使用 Isaac Sim v6.0.1 上游示例，保留 Franka 和 Nova Carter 的关节控制逻辑。
`--record` 录制视口并添加轮次与步数，60 Hz 物理以 30 FPS 半速播放。

当前 Gemini 会话确认 MY_DFS、选择空闲 ISAAC_SIM_GPU 后运行：

```bash
bash examples/isaac_sim_tutorial/test_getting_started_robot.sh \
  --record "$RUN_DIR/getting_started_robot.mp4" "--/log/file=$RUN_DIR/kit.log"
```

本次验证只检查教程运行和视频，不对机器人控制精度做数值断言。

## 2026-10-26 运行与录像

任务 `9a1e3286-0012` 退出 0，耗时约 223 秒，其中约 191 秒用于应用启动与着色器编译。
四轮日志完整，最后输出 `Recorded 400 frames`，无 Python traceback。
Franka 与 Nova Carter 均可见，四轮抽帧确认关节姿态变化、Carter 移动和轮次标签。

环境：Isaac Sim 6.0.1.0、Python 3.12.13、GPU 1 NVIDIA RTX PRO 5000 72GB Blackwell。
Docker image tag、容器内 Isaac Sim 构建 commit：未记录。
Carrot 基线 commit：`a49a44cf3019ccb6088466bbd3369557a0ef35eb`，本次改动尚未提交。
上游源码：v6.0.1，commit `045ca8b59622b99a408092124377c66346e8d9c2`。

实际命令（MY_DFS 来自本次 Gemini 会话，资产为已有本地 6.0 资源）：

```bash
export MY_DFS=/mnt/ceph-zjk1-csp/mm-base-plt2/nrwu
export ISAAC_SIM_GPU=1
RUN_DIR="$MY_DFS/tests/carrot/isaac_sim_tutorial/robot-recording-20261026-03"
bash examples/isaac_sim_tutorial/test_getting_started_robot.sh \
  --record "$RUN_DIR/getting_started_robot.mp4" "--/log/file=$RUN_DIR/kit.log" \
  "--/persistent/isaac/asset_root/default=$MY_DFS/isaacsim_assets/Assets/Isaac/6.0"
```

证据目录为上述 RUN_DIR，包含 MP4、kit.log、run.log。
本地视频：`~/Downloads/isaac_sim_tutorial/getting_started_robot-20261026.mp4`。
预览：同目录 `getting_started_robot-20261026-preview.png`。
ffprobe 验证：H.264、1280×720、30 FPS、400 帧、13.333333 秒、521218 字节；
ffmpeg 全片解码通过，本地与远端视频 SHA256 一致。

SHA256：

- Python：`9613e3487d1c7b0b03da207a76a4cdab5aac22af74cc40ff0f6c50affbc4f271`
- shell：`3387e7c04517659b39baaeda4db400eceb5ad9d0ec0972f21a94589c55f4d9d6`
- MP4：`ec9b3e8cf867054fe29965f636e4a0da19e3e7b43523d27af154253dff798382`

Python 语法、bash -n、ruff check、ruff format --check 和 git diff --check 通过。
首次两次运行在第一轮遇到 RGB 空帧断言；只调整预热仍失败，最终增加显式 Replicator
采集（delta_time=0，pause_timeline=False）后录制成功。失败日志分别保存在
`robot-recording-20261026-01` 和 `robot-recording-20261026-02`，未覆盖。

环境告警已向用户报告：Kit 辅助进程启动报 Resource temporarily unavailable，
无显示器及音频设备告警仍存在；未安装、升级依赖或修改环境配置。
录制结束后确认仿真进程已退出，并恢复 dguard：DGUARD_WATCH=1，无延迟恢复任务。
