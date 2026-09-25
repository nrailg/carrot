# 实验 recipes

`recipes/` 记录正式训练和评测实验；`tests/` 记录代码与数据契约的验证。
每个实验使用独立目录，保留可复用的运行脚本、配置路径、参数、运行状态和证据位置。
历史实验只记录有证据的结果，计划步数和实际完成步数分开写；重跑结果追加到同一实验档案。

| 实验 | 配置 | 记录 |
| --- | --- | --- |
| PI0.5 + LIBERO 2k SFT | [`pi05_sft_libero_2k_20260919.yaml`](pi05_libero_sft_2k_20260919/pi05_sft_libero_2k_20260919.yaml) | [`pi05_libero_sft_2k_20260919/README.md`](pi05_libero_sft_2k_20260919/README.md) |
| PI0.5 + SO101 orange cube SFT | [`pi05_sft_so101_orange_cube.yaml`](pi05_sft_so101_orange_cube/pi05_sft_so101_orange_cube.yaml) | [`pi05_sft_so101_orange_cube/README.md`](pi05_sft_so101_orange_cube/README.md) |
