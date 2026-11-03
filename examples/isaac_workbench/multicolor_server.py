# ruff: noqa: E402, I001
import argparse
import json
import os
import traceback
from multiprocessing.connection import Listener

from multicolor_controls import remove_sensitive_environment

parser = argparse.ArgumentParser()
parser.add_argument("--socket", required=True)
parser.add_argument("--config", required=True)
parser.add_argument("--gpu", type=int, required=True)
args = parser.parse_args()
remove_sensitive_environment()
os.sched_setaffinity(0, sorted(os.sched_getaffinity(0))[:16])
os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
physics_gpu = visible.split(",").index(str(args.gpu)) if visible else args.gpu

from isaacsim import SimulationApp

app = SimulationApp(
    {
        "headless": True,
        "active_gpu": args.gpu,
        "physics_gpu": physics_gpu,
        "multi_gpu": False,
        "limit_cpu_threads": 16,
        "enable_crashreporter": False,
        "extra_args": [
            "--/plugins/carb.tasking.plugin/threadCount=16",
            "--/telemetry/enableAnonymousData=false",
            "--/telemetry/enableSentry=false",
        ],
    }
)

from multicolor_env import MulticolorEnv

env = MulticolorEnv(**json.loads(args.config))
listener = Listener(args.socket, family="AF_UNIX")
connection = listener.accept()
connection.send({"ready": True})
try:
    while True:
        method, kwargs = connection.recv()
        if method == "close":
            connection.send({"ok": True, "result": None})
            break
        methods = {
            "reset": env.reset,
            "step": env.step,
            "observe": env.observe,
            "snapshot": env.snapshot,
            "check_reset": env.check_reset,
            "collect": env.collect,
            "expert_actions": env.expert_actions,
        }
        assert method in methods, f"unsupported simulator command: {method}"
        try:
            result = methods[method](**kwargs)
        except BaseException:
            connection.send({"ok": False, "traceback": traceback.format_exc()})
            raise
        connection.send({"ok": True, "result": result})
finally:
    env.close()
    connection.close()
    listener.close()
    app.close()
