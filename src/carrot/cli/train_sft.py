"""Run supervised fine-tuning from a YAML config."""

from __future__ import annotations

import argparse

from carrot.trainer.sft import SFTConfig, run_sft


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a LeRobot policy with Carrot FSDP2")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    run_sft(SFTConfig.from_yaml(args.config), resume=args.resume)


if __name__ == "__main__":
    main()
