# SO101 frame-prompt CPU契约测试

## 目的与预期

保护唯一共享formatter的原task尾空白处理、非负整数与四位最小宽度（10000合法）、确定性与非法prompt/index拒绝。
训练/推理相同index生成完全一致prompt；重复取样或source.read不累计suffix。训练包装仅修改
prompt，原样本/state/action/夹爪/图像不被其修改；state jitter仍遵守五轴限位，统计量保持原引用。
factory拒绝多episode与长度不匹配。

fake policy验证客户端无图dict不被修改，服务本地适配先补零腕图满足SO101Inputs，
实际完整input transform再复用DropVision，所有image mask=false、pixels=0，
prompt保持完整；无需加载checkpoint或运行模型。握手必须同时具有frame_index_prompt=true与
no_vision=true，启用帧prompt的客户端拒绝普通vision服务，而普通模式保持原握手契约。

fake dataset/robot/policy验证h10/K1与K5、重复read/预热/反馈读取不推进、尾块不足K及N处停止；
显式起点生效，普通模式保留原prompt和真实图像映射。无视觉明确base_camera/wrist_camera均为null，
客户端只有state/prompt，不生成图像、不依赖相机观测；SDK连接工厂收到空camera配置。
帧prompt与camera配置正交，0/1/2个视角均可独立配置；真机cameras匹配校验不能因帧prompt绕过。
总帧数10001合法，仍要求正整数、起点合法。所有机器人构造/连接均使用替身，
不访问串口、真实相机、GPU或模型服务。

未来在已授权远端环境、已确认MY_DFS并同步源码后执行：

```bash
bash tests/so101_real/test_so101_frame_prompt.sh
```

三件套：同目录 `test_so101_frame_prompt.py`、`.sh`、`.md`。

## 2026-10-09

状态：**NOT RUN / PENDING**。按本轮约束，仅Mac编写和静态检查，未执行正式pytest、远程操作、
训练、模型推理、服务或硬件。静态检查不能替代pytest通过。
实际Docker image tag、测试Carrot commit与上游commit：未记录（尚未运行）。
本地静态检查（exit 0）：

- `python -m compileall -q`：共享formatter、五个修改的客户端模块、新recipe augmentation/serve及测试。
- `bash -n recipes/pi05_sft_so101_knock_down_the_cylinder_frame_prompt/run.sh tests/so101_real/test_so101_frame_prompt.sh`。
- `git diff --check`。
- `rg`核对本轮目录不存在旧renderer、图模式键与旧recipe引用。

本地轻量fake smoke：PASS（直接调用examples纯Python模块，非pytest）。验证formatter接受10000、
拒绝非法index，帧条件开/关与0/1/2视角配置独立；fake DatasetSource/RobotSource在无图片字段下
请求只有state/prompt，K1/K5推进7帧并终止，重复read不推进；10000起点/10001总帧数末块1步后停止。
轻量模块未导入carrot/Ray。服务适配与完整pytest尚待正式测试。

Ruff未执行：本地未找到ruff命令。没有GPU/真机效果结论。
