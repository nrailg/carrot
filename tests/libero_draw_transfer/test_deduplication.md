# 示范发布前去重检查

命令：`bash tests/libero_draw_transfer/test_deduplication.sh`。

检查64条成功三色episode的seed、完整actions.npy和state.npy内容SHA分别唯一；不同文件名或seed不能掩盖拷贝轨迹。仅episode级去重，不删除合理的相邻相似帧。发布时写deduplication.json，失败不发布到训练目录。

2026-10-24 H20：4项PASS，0.11秒。代码四文件SHA已核验，冻结于个人DFS的`experiments/carrot/pi05_libero_draw_sft_transfer/sources/deduplication-20261024-01`。当前64条正式数据尚未生成，测试通过不等于实际数据已去重验收。

基线6301541；Docker image和上游commit未记录。
