import argparse
import logging
import os
from pathlib import Path

from .config import load_config
from .runner import run


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug and deploy a Carrot SO101 policy")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--observation-source", choices=("dataset", "robot"))
    parser.add_argument("--action-sink", choices=("log", "robot"))
    parser.add_argument("--server-uri")
    parser.add_argument("--dataset-root")
    parser.add_argument("--output-dir")
    parser.add_argument("--episode", type=int)
    parser.add_argument("--start-frame", type=int)
    parser.add_argument("--execute-steps", type=int)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--prompt")
    args = vars(parser.parse_args())
    path = args.pop("config")

    # 离线标志在加载本地数据集前设置，避免依赖库尝试访问 Hub。
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["HF_DATASETS_OFFLINE"] = "1"

    logging.basicConfig(level=logging.INFO)
    run(load_config(path, args))


if __name__ == "__main__":
    main()
