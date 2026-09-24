# 验证记录

日期：2026-09-23。范围仅限本目录，未修改既有环境实现。

## 已完成

在本地开发机执行：

```bash
ruff check examples/isaac_custom_env
ruff format --check examples/isaac_custom_env
python - <<'PY'
import ast
from pathlib import Path
for path in sorted(Path("examples/isaac_custom_env").glob("*.py")):
    ast.parse(path.read_text(), filename=str(path))
    print("AST PASS", path)
PY
git diff --check
```

- Python 3.12：4 个 Python 文件语法解析通过；没有导入执行示例模块。
- Ruff 0.16.7：lint 与格式检查通过。
- 官方教程链接：从本目录 Python/Markdown 中提取 19 个唯一的
  `isaac-sim.github.io` URL，用 `urllib.request.urlopen(timeout=15)` 发出 GET；19/19 返回 200。
  修复了 Arena 入门页已从 `first_arena_env.html` 移到 `arena_env.html` 的引用。
- 静态接口核对：对照 SOURCES.md 的固定上游源码，检查了启动顺序、配置字段、关节动作语义、
  reach 距离函数、自动 reset、Arena TaskBase 必需方法与 Builder 组合路径。
- 主模型独立复查了语法/lint/格式与教程 URL；未仅依据子代理的通过报告。

## 尚未完成

当前 Python 的 `importlib.util.find_spec("isaaclab")` 返回 `None`。
没有安装依赖或运行 GPU 仿真，因此没有以下证据：

- Panda USD 成功加载、场景画面/物理稳定性。
- 实际 action/observation shape、成功和超时路径。
- 两条入口 Lab/Arena 的实机一致性。
- SO101 USD 导入、驱动参数、可达性和任务成功。
- 相机、遥操作、学习器训练和最终观测/timeout bootstrap 适配。

README 的命令是后续运行入口，配置推导的 shape/时步是预期值，不是已通过的 GPU 结果。
`custom_arm.py` 是资产接入 helper；`SO101.md` 中的参数化片段需要填入具体资产信息。

## 2026-09-23：按用户要求委派 Luna 做远程验证

已准备 [GPU 合同检查与执行记录](../../tests/isaac_custom_env/test_custom_env.md)。
Luna 及主模型分别尝试连接用户指定的 Gemini launcher，均返回
`Failed to connect to remote container`，未获得活动会话。
当前状态为 **BLOCKED / GPU NOT RUN**；尚未同步或执行，不能将本例标记为 GPU PASS。
测试恢复后在上述同一档案追加实际命令、版本和结果。

## 2026-09-23：新节点资产及测试状态

新 launcher 已连接。当前个人 CephFS 为
`/mnt/ceph-zjk1-csp/mm-base-plt2/nrwu`。已将 Panda、Grid、UIElements
资产子树从另一块个人 CephFS 同步到当前个人 CephFS，按文件内容比较无差异。
示例现在支持 `--asset_root`，以同一个本地资产根同时指定 Panda、地面和目标坐标轴 USD。

Luna 的后台任务 `11bf11e8-0019` 在启动 Isaac 前因 CephFS 结果目录创建权限失败，
退出码为 1。远程命令以 root 运行，个人目录属于 UID 1001、权限 755；
由于挂载是 `fuse.dop-fuse` 且 root 具有 `CAP_DAC_OVERRIDE`，仅靠这组传统
权限位无法解释拒绝，具体的 FUSE/后端策略尚未确认。
**Lab/Arena GPU 用例仍是 NOT RUN**，不能据此评价示例运行结果。错误原文、
环境版本、目标结果目录和复现矩阵见
[测试记录](../../tests/isaac_custom_env/test_custom_env.md)。
暂停的本节点 dguard 已恢复，并独立检查其巡检与运行进程状态。
