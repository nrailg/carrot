"""Create an isolated run configuration and normalization copy without starting training."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import yaml
from yaml.nodes import MappingNode, Node, ScalarNode, SequenceNode

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
NORM_SHA256 = "ed6087fd3e59c7e103a295abe2811241d7b30f5ae88975258d348e4a747dac0b"
_PLACEHOLDER = re.compile(r"\$\{([A-Z_]+)\}")


def _render_template(template: str, paths: dict[str, str]) -> str:
    """Replace parsed YAML scalar values while retaining source comments and safe quoting."""
    replacements: list[tuple[int, int, str]] = []
    used: set[str] = set()

    def visit(node: Node) -> None:
        if isinstance(node, ScalarNode) and _PLACEHOLDER.search(node.value):

            def replace(match: re.Match[str]) -> str:
                name = match[1]
                if name not in paths:
                    raise ValueError(f"Unknown configuration placeholder: {name}")
                used.add(name)
                return paths[name]

            value = _PLACEHOLDER.sub(replace, node.value)
            replacements.append((node.start_mark.index, node.end_mark.index, json.dumps(value)))
        elif isinstance(node, MappingNode):
            for key, value in node.value:
                visit(key)
                visit(value)
        elif isinstance(node, SequenceNode):
            for value in node.value:
                visit(value)

    document = yaml.compose(template, Loader=yaml.SafeLoader)
    if document is None:
        raise ValueError("The recipe template is empty")
    visit(document)
    if used != paths.keys():
        raise ValueError(f"Missing configuration placeholders: {sorted(paths.keys() - used)}")
    for start, end, value in sorted(replacements, reverse=True):
        template = template[:start] + value + template[end:]
    yaml.safe_load(template)
    return template


def configure(data_root: Path, model_path: Path, tokenizer_path: Path, run_dir: Path) -> Path:
    """Write train.yaml, a private norm copy, and provenance in a new absolute run directory.

    The data root must exist. The run directory must not exist or be inside the source
    checkout or any input directory. Model paths may target the training host; their
    contents are checked by the separate read-only preflight. Returns the config path.
    """
    named_paths = {
        "data_root": data_root,
        "model_path": model_path,
        "tokenizer_path": tokenizer_path,
        "run_dir": run_dir,
    }
    for name, path in named_paths.items():
        if not path.is_absolute():
            raise ValueError(f"{name} must be an absolute path: {path}")
    if run_dir.exists() or run_dir.is_symlink():
        raise FileExistsError(f"Refusing to overwrite run directory: {run_dir}")
    data_root, model_path, tokenizer_path, run_dir = (
        path.resolve() for path in named_paths.values()
    )
    if not data_root.is_dir():
        raise NotADirectoryError(f"data_root must be an existing directory: {data_root}")
    for protected in (REPO_ROOT, data_root, model_path, tokenizer_path):
        if run_dir.is_relative_to(protected):
            raise ValueError(
                f"run_dir must be outside the source and input directories: {protected}"
            )

    norm = (HERE / "assets/pi05_norm_stats.json").read_bytes()
    if hashlib.sha256(norm).hexdigest() != NORM_SHA256:
        raise ValueError("The recipe joint normalization SHA256 does not match the fixed release")
    template = (HERE / "train.template.yaml").read_text()
    rendered = _render_template(
        template,
        {
            "DATA_ROOT": str(data_root),
            "MODEL_PATH": str(model_path),
            "TOKENIZER_PATH": str(tokenizer_path),
            "RUN_DIR": str(run_dir),
            "NORM_PATH": str(run_dir / "assets/pi05_norm_stats.json"),
        },
    )
    provenance = {
        "created_at": datetime.now(UTC).isoformat(),
        "source_root": str(REPO_ROOT),
        "recipe_root": str(HERE),
        "template_sha256": hashlib.sha256(template.encode()).hexdigest(),
        "inputs_sha256": hashlib.sha256((HERE / "inputs.json").read_bytes()).hexdigest(),
        "norm_sha256": NORM_SHA256,
        "config_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
    }
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "assets").mkdir()
    norm_path = run_dir / "assets/pi05_norm_stats.json"
    norm_path.write_bytes(norm)
    norm_path.chmod(0o444)
    config_path = run_dir / "train.yaml"
    config_path.write_text(rendered)
    (run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return config_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--tokenizer-path", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print(configure(args.data_root, args.model_path, args.tokenizer_path, args.run_dir))


if __name__ == "__main__":
    main()
