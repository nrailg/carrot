# SO101 recording to SFT

Prepare the [environments](environments.md), then follow the steps below to collect
data and hand off an SFT experiment. Single-trajectory fitting can check the
training pipeline; task experiments measure success and performance on unseen
initial conditions.

## 1. Prepare the hardware and task

Identify the leader/follower serial ports, calibrated robot IDs, calibration
files, and wrist-camera index. Check teleoperation, continuous camera capture,
starting poses, joint ranges, and cable clearance before recording. Device
settings must match the actual hardware.

Additional local hardware notes are available in the neighboring practices
repository's [LeRobot SO101 tutorial](../../practices/posts/tools/lerobot.md).
That tutorial is not distributed with Carrot. Its older recording example has
inline comments after shell continuation characters; use the command below.

For the first dataset, use one task with fixed camera placement, objects, and
lighting. Define success and the allowed variation in object starting positions.
The example task is knocking down an upright cylinder; record the full
demonstration and a final state that makes its outcome visible.

## 2. Record locally

Activate the recording environment and configure the actual devices:

```bash
source "$HOME/venvs/lerobot-0.6.1/bin/activate"
export FOLLOWER_PORT=/dev/tty.usbmodem_FOLLOWER
export LEADER_PORT=/dev/tty.usbmodem_LEADER
export FOLLOWER_ID=my_awesome_follower_arm
export LEADER_ID=my_awesome_leader_arm
export WRIST_CAMERA_INDEX=0
```

Replace the port placeholders, IDs, and camera index above. The following command
starts teleoperation and records **two episodes**, each lasting at most
**120 seconds**, with up to **60 seconds** for manual resets. Press `n` to finish
the current phase early.

```bash
lerobot-record \
  --robot.type=so101_follower \
  --robot.port="$FOLLOWER_PORT" \
  --robot.id="$FOLLOWER_ID" \
  --robot.use_degrees=true \
  --robot.cameras="{wrist: {type: opencv, index_or_path: $WRIST_CAMERA_INDEX, width: 640, height: 480, fps: 15, warmup_s: 5}}" \
  --teleop.type=so101_leader \
  --teleop.port="$LEADER_PORT" \
  --teleop.id="$LEADER_ID" \
  --teleop.use_degrees=true \
  --dataset.repo_id=local/knock_down_the_cylinder \
  --dataset.single_task="Knock down the cylinder" \
  --dataset.num_episodes=2 \
  --dataset.episode_time_s=120 \
  --dataset.reset_time_s=60 \
  --dataset.fps=15 \
  --dataset.push_to_hub=false \
  --display_data=false
```

Right arrow / `n` finishes the current phase, left arrow / `r` rerecords the
current episode, and Esc / `q` stops. Terminal keyboard mode requires terminal
focus. During reset, teleoperation continues but frames are not saved: reset the
object and use the leader to return the follower to its starting pose. There is
no automatic homing.

The five motion joints use degrees; the gripper uses opening percentage.
`action` is the calibrated leader target, while `observation.state` is the
follower's measured position. They are different signals.
This example names the camera `wrist` and uses 15 FPS for recording and deployment.
The installed LeRobot 0.6.1 recording workflow appends a timestamp to new dataset
names; use the actual output path printed by the recorder.

## 3. Validate and transfer the dataset

Check a small trial batch before collecting more episodes:

- Verify complete `meta/`, `data/`, and `videos/` directories, episode counts,
  frame counts, and FPS.
- Review every video for valid wrist images, a complete task, dropped frames,
  long idle periods, and incorrect views.
- Check finite six-dimensional state/action values, joint order, units, task
  text, and camera keys.
- Record successful, failed, and discarded episodes and their reasons. Review
  failed demonstrations before mixing them into a successful-demonstration set.

Use the [dataset inspection script](../examples/lerobot/inspect_dataset.py).
Its [shell wrapper](../examples/lerobot/inspect_dataset.sh) contains historical
paths that must be updated. Split training and validation by episode rather
than splitting adjacent frames from the same demonstration.

Copy the complete dataset to personal persistent storage accessible from the
GPU server. Verify metadata and file checksums, then sample video decoding on
the server. Transfer the matching calibration JSON when an experiment needs it.
Keep datasets, videos, weights, and runtime logs outside the code repository.

## 4. Train and evaluate

See [PI0.5 SFT](pi05-sft.md) for training entry points, statistics, and checkpoints.
Copy the [cylinder wrist-camera configuration](../recipes/pi05_sft_so101_knock_down_the_cylinder_overfit/train.yaml)
and update dataset ID/root, model/tokenizer paths, output directory, training
budget, and GPU layout. Historical runners also contain fixed paths; editing
only the YAML is insufficient when reusing those runners.

The [Arm Dance](../recipes/pi05_sft_so101_arm_dance/README.md) and
[official LeRobot comparison](../recipes/lerobot_pi05_arm_dance/README.md) recipes
are single-episode diagnostics with vision disabled. Use a visual task
configuration for new visual demonstrations; historical diagnostic results do
not establish performance on the new dataset.

After training, check checkpoint loading and offline predictions, then follow
the [deployment guide](../examples/so101_real/README.md) to validate live inputs
and execution. Record trial counts, successes, initial object positions, failure
reasons, and videos. Evaluate training-covered and unseen positions separately.
Report model action error, servo tracking error, and task success separately.

Each handoff includes the task and success criterion, hardware/calibration/camera
settings, dataset location and split, quality checks, source version, effective
training configuration, checkpoint path, offline metrics, and robot trial records.
