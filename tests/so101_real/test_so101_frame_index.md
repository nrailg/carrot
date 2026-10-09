# SO101 frame-index image CPU契约测试

## 目的与预期

保护唯一七段renderer的RGB uint8[224,224,3]、确定性、不同index可辨认及非法输入拒绝；
训练和推理相同index逐像素一致，原样本/state/action/夹爪不被图片替换修改，state jitter仍遵守
五轴限位，统计量保持原引用。factory拒绝多episode与长度不匹配，SO101Inputs仅wrist mask为真。

fake dataset/robot/policy验证h10/K1与K5、重复read/预热/反馈读取不推进、尾块不足K及N处停止；
显式起点生效，普通模式保留原真实图像映射，合成模式不依赖相机观测且SDK连接工厂收到空camera
配置。所有机器人构造/连接均使用替身，不访问串口、真实相机、GPU或模型服务。

未来在已授权远端环境、已确认MY_DFS并同步源码后执行：

```bash
bash tests/so101_real/test_so101_frame_index.sh
```

三件套：同目录 `test_so101_frame_index.py`、`.sh`、`.md`。

## 2026-10-09

状态：**NOT RUN / PENDING**。按本轮约束，仅Mac编写和静态检查，未执行正式pytest、远程操作、
训练、模型推理、服务或硬件。编译、bash语法与纯renderer smoke不能替代pytest通过。
实际Docker image tag、测试Carrot commit与上游commit：未记录（尚未运行）。
本地静态检查（exit 0）：

- `python -m compileall -q`：新增renderer、recipe augmentation、测试，以及修改的五个客户端模块。
- `bash -n recipes/pi05_sft_so101_arm_dance_frame_index/run.sh tests/so101_real/test_so101_frame_index.sh`。
- `git diff --check`。
- 纯renderer smoke：0/143/9999 shape/dtype/确定性、143与144图区别；非法输入拒绝。

Ruff未执行：本地未找到ruff命令。没有GPU/真机效果结论。
