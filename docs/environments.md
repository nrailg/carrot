# Environments

Carrot requires Python **3.12**. Start with the tools below, then install only
the environments needed for your task. No existing venv or container environment
is required. Commands use a POSIX shell on macOS or Linux.

## Install the tools

Run the numbered commands in each cell in order. Skip a tool only after checking
that it is already installed.

| Tool | Installation | Check |
| --- | --- | --- |
| Git and basic build tools | macOS: `xcode-select --install`, then complete the installer.<br>Ubuntu/Debian: `sudo apt-get update`, then `sudo apt-get install -y git curl ca-certificates build-essential`. | `git --version` |
| uv | `curl -LsSf https://astral.sh/uv/install.sh \| sh`<br>Open a new terminal, or run `source "$HOME/.local/bin/env"` for the default installation. See the [official installer](https://docs.astral.sh/uv/getting-started/installation/) for other platforms and install locations. | `uv --version` |
| Python 3.12 | `uv python install 3.12` | `uv python find 3.12` |

Clone the repository with `git clone https://github.com/nrailg/carrot.git`, then
run `cd carrot`. All paths and commands below are relative to that repository
root unless stated otherwise.

## Install an environment

Run each row's installation commands in order, then activate that environment.
These are installation instructions, not commands to run training or move a robot.

| Purpose / platform | Environment | Create and install | Activate |
| --- | --- | --- | --- |
| CPU runtime demo; macOS or Linux | `.venv` | 1. `uv venv --python 3.12 .venv`<br>2. `uv --no-config pip install --python .venv/bin/python -e ".[dev]"` | `source .venv/bin/activate` |
| SO101 teleoperation and recording; macOS | `~/venvs/lerobot-0.6.1` | 1. `uv venv --python 3.12 "$HOME/venvs/lerobot-0.6.1"`<br>2. `uv --no-config pip install --python "$HOME/venvs/lerobot-0.6.1/bin/python" --torch-backend cpu 'lerobot[core_scripts,feetech]==0.6.1'` | `source "$HOME/venvs/lerobot-0.6.1/bin/activate"` |
| SO101 inference client; macOS | `examples/so101_real/.venv` | 1. `uv venv --python 3.12 examples/so101_real/.venv`<br>2. `uv --no-config pip install --python examples/so101_real/.venv/bin/python --torch-backend cpu --overrides examples/so101_real/overrides.txt -r examples/so101_real/requirements.txt` | `source examples/so101_real/.venv/bin/activate` |
| PI0.5 training and serving; Linux x86_64 with an NVIDIA GPU | `.venv` | 1. `uv venv --python 3.12 .venv`<br>2. `uv sync --frozen --no-install-project --extra sft --extra serving --extra dev` | `source .venv/bin/activate` |
| Official LeRobot comparison; Linux x86_64 with an NVIDIA GPU, optional | `~/venvs/lerobot-official-arm-dance` | 1. `uv venv --python 3.12 "$HOME/venvs/lerobot-official-arm-dance"`<br>2. `uv --no-config pip install --python "$HOME/venvs/lerobot-official-arm-dance/bin/python" --torch-backend cu128 'lerobot[pi,training,async]==0.6.1' 'torch==2.11.0' 'torchvision==0.26.0' 'transformers==5.5.4' 'accelerate==1.15.0'` | `source "$HOME/venvs/lerobot-official-arm-dance/bin/activate"` |
| Isaac Sim / Lab examples; Linux x86_64 with a supported NVIDIA GPU, optional | `examples/isaac/.venv` | 1. `uv sync --project examples/isaac --python 3.12 --frozen --no-group arena`<br>2. For Arena example 03, run `uv sync --project examples/isaac --frozen --group arena` instead. | `source examples/isaac/.venv/bin/activate` |

GPU rows require a host NVIDIA driver compatible with the selected CUDA packages;
the Carrot lockfile selects CUDA 12.8 PyTorch wheels. Check `nvidia-smi` on the
host first. A venv installs Python packages, not a GPU driver. On managed servers,
select a compatible GPU image or ask the platform administrator to prepare the
driver. Isaac has additional hardware and asset requirements described in the
[Isaac guide](../examples/isaac/README.md).

The CPU demo and GPU training rows are alternative setups for the root `.venv`;
choose the row matching the machine. The GPU installation uses dependencies from
[pyproject.toml](../pyproject.toml) and the committed lockfile, without installing
Carrot itself.
The root uv configuration targets Linux x86_64 / CUDA, which is why Mac installs
use `--no-config` and separate requirements.

## Load the source and check the setup

For the Mac inference client and Carrot GPU environment, activate the chosen
environment, then run this from the repository root in each new shell:

```bash
export PYTHONPATH="$PWD/src:$PWD:${PYTHONPATH:-}"
```

| Environment | Check / next step |
| --- | --- |
| CPU runtime demo | Run `python examples/distributed_smoke.py`; see the [demo](../examples/distributed_smoke.py). |
| Mac recording | Check `lerobot-record --help` and `lerobot-teleoperate --help`, then follow the [recording guide](so101-data-sft.md) for calibration and devices. |
| Mac inference client | Run `python -m examples.so101_real.main --help`; continue with [deployment](../examples/so101_real/README.md). The [requirements](../examples/so101_real/requirements.txt) and [NumPy overrides](../examples/so101_real/overrides.txt) are specific to this client. |
| Carrot GPU | Run `uv pip check --python .venv/bin/python` and `python -m carrot.cli.train_sft --help`; prepare model weights, tokenizer, data, and Ray resources using the [SFT guide](pi05-sft.md). |
| Official LeRobot comparison | Run `uv pip check --python "$HOME/venvs/lerobot-official-arm-dance/bin/python"`. The [historical recipe](../recipes/lerobot_pi05_arm_dance/README.md) pins a different venv path and existing data/model paths; adapt those before reusing its runner. The install above does not reconstruct every historical package version. |
| Isaac examples | Continue with the [Isaac guide](../examples/isaac/README.md) to prepare assets and choose an example. The Lightwheel example uses a separate environment. |

Calibration files, datasets, model weights, and simulator assets are separate
from venv installation. Recording dependencies provide the CLI, but do not create
hardware calibration or demonstrations.

## Existing GPU containers

Some Carrot images already provide `/opt/venvs/carrot`. Use it only after verifying
that `/opt/venvs/carrot/bin/python` exists and the image is the intended one.
Activate it, change to the synchronized repository, and set `PYTHONPATH` as above.
Otherwise use the fresh Linux GPU installation in the table.

Run commands directly with the activated Python. Project `uv run` or an unqualified
`uv sync` selects a project environment rather than automatically using an
activated container venv. Do not sync into a shared preinstalled environment.
Historical runners that explicitly activate `/opt/venvs/carrot` must be adapted
when using the new `.venv`.
