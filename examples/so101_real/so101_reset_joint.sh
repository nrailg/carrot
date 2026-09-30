source .venv/bin/activate

SO101_PY=python
SO101_SET=so101_set_joint.py

# 1：底座
"$SO101_PY" "$SO101_SET" --joint 1 --target 0

# 2：肩部
"$SO101_PY" "$SO101_SET" --joint 2 --target -100

# 3：肘部，分两次
"$SO101_PY" "$SO101_SET" --joint 3 --target 90

# 4：手腕俯仰
"$SO101_PY" "$SO101_SET" --joint 4 --target 60

# 5：手腕旋转
"$SO101_PY" "$SO101_SET" --joint 5 --target 0

# 6：夹爪
"$SO101_PY" "$SO101_SET" --joint 6 --target 0
