#!/usr/bin/env bash
set -Eeuo pipefail

: "${MY_DFS:?set MY_DFS from the current Gemini session}"
: "${ISAAC_SIM_GPU:?select an idle GPU}"
source /opt/venvs/carrot/bin/activate
cd "${MY_DFS}/work/carrot"

# Kit 辅助进程可能打印环境，只传运行必需项，避免把会话凭据写入日志。
exec env -i HOME="$HOME" PATH="$PATH" LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}" \
  VK_ICD_FILENAMES="${VK_ICD_FILENAMES:-/etc/vulkan/icd.d/nvidia_icd.json}" \
  OMNI_KIT_ACCEPT_EULA=YES \
  python -u examples/isaac_sim_tutorial/getting_started_robot.py --no-window \
  "--/renderer/activeGpu=${ISAAC_SIM_GPU}" "--/physics/cudaDevice=${ISAAC_SIM_GPU}" \
  --/renderer/multiGpu/enabled=false \
  --/plugins/carb.tasking.plugin/threadCount=16 \
  --/plugins/omni.tbb.globalcontrol/maxThreadCount=16 \
  --/telemetry/enableAnonymousData=false "$@"
