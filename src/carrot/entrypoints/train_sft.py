"""Run supervised fine-tuning from a YAML config."""

from __future__ import annotations

import argparse

from carrot.trainer.sft import SFTConfig, SFTTrainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a LeRobot policy with Carrot FSDP2")
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume")
    args = parser.parse_args()
    SFTTrainer(SFTConfig.from_yaml(args.config), resume=args.resume).run()


if __name__ == "__main__":
    main()
