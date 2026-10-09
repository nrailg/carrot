import argparse
from dataclasses import replace
from pathlib import Path
from typing import Any

from carrot.models.pi05 import transforms
from carrot.models.pi05.inference import create_so101_policy
from carrot.models.pi05.inference.websocket_policy_server import WebsocketPolicyServer
from recipes.pi05_sft_so101_fit_validation.fit import DropVision


def enable_no_vision(policy: Any) -> None:
    """Apply the recipe's shared DropVision transform to an inference policy.

    Parameters
    ----------
    policy : Any
        PI0.5 policy exposing its transform spec and composed input transform.

    Returns
    -------
    None
        Updates the policy input pipeline in place; does not load or run the model.
    """
    spec = replace(policy._transform_spec, inputs=policy._transform_spec.inputs + (DropVision(),))
    policy._transform_spec = spec
    policy._input_transform = transforms.compose(spec.inputs)


def main() -> None:
    parser = argparse.ArgumentParser(description="Serve the SO101 frame-prompt no-vision recipe")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--num-steps", type=int, default=10)
    args = parser.parse_args()
    policy = create_so101_policy(args.checkpoint, device=args.device, num_steps=args.num_steps)
    enable_no_vision(policy)
    WebsocketPolicyServer(
        policy, host=args.host, port=args.port,
        metadata={**policy.metadata, "embodiment": "so101", "no_vision": True,
                  "frame_index_prompt": True},
    ).serve_forever()


if __name__ == "__main__":
    main()
