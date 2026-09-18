"""Serve a Carrot PI0.5 policy over the OpenPI WebSocket protocol."""

from __future__ import annotations

import argparse

from carrot.models.pi05.inference import create_libero_policy, create_robotwin_policy
from carrot.models.pi05.inference.websocket_policy_server import WebsocketPolicyServer


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve a Carrot PI0.5 checkpoint")
    parser.add_argument("--embodiment", choices=("robotwin", "libero"), required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--tokenizer-path")
    parser.add_argument("--norm-stats-path")
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--default-prompt")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()

    factory = create_robotwin_policy if args.embodiment == "robotwin" else create_libero_policy
    policy = factory(
        args.checkpoint,
        device=args.device,
        tokenizer_path=args.tokenizer_path,
        norm_stats_path=args.norm_stats_path,
        num_steps=args.num_steps,
        default_prompt=args.default_prompt,
    )
    WebsocketPolicyServer(
        policy,
        host=args.host,
        port=args.port,
        metadata=policy.metadata,
    ).serve_forever()


if __name__ == "__main__":
    main()
