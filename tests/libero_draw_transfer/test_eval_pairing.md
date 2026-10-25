# 配对评测noise契约

验证同一episode/调用号的noise与执行顺序、其他模型停止时间无关；不同episode和调用号使用不同noise。沿用Pi05Policy公开noise输入，ODE10，execute5。运行test_eval_pairing.sh；预期2PASS，真实checkpoint加载和闭环另由正式评测提供证据。

状态：PASS，H20最终检查task b434d471-0189，2项/0.13秒，ruff通过；两池123文件SHA一致。开发基线30392d7（先前训练基线6301541加mask修复），评测接线未提交；Docker和上游commit未记录。
