import argparse

from carrot.trainer.rl.config import load_config
from carrot.trainer.rl.trainer import run


def main() -> None:
    parser = argparse.ArgumentParser(description="PI0.5 multicolor tracing RL transfer pilot")
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--resume", help="trusted learner checkpoint; starts fresh simulator episodes"
    )
    args = parser.parse_args()
    run(load_config(args.config), args.resume)


if __name__ == "__main__":
    main()
