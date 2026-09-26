# PI0.5 LIBERO SFT train/inference alignment

## Contract

The LeRobot LIBERO adapter returns inference-style raw state, two images and
task prompt, plus a 10-frame 7D action chunk and its padding mask. The dataset
applies the exact LIBERO inference input transforms to each sample before
collation. `Pi05SFTLossFn` consumes the transformed batch, retaining its
existing noise/time sampling and valid-step loss reduction. The RobotWin path
remains unchanged.

The real-sample test compares the same sample at train and inference entry
points: images, masks, state, token IDs and token mask must be exact. It checks
the quantile-normalized `(10, 32)` target independently against official stats
and performs one finite forward loss without training steps.
The configured stats file must take precedence over other stats. If file stats
are selected but the file is missing, model setup must fail instead of using
dataset stats.

## Run

Resolve `MY_DFS` in the active Gemini session and sync this checkout. Confirm
local checkpoint, tokenizer and dataset assets, plus dguard state, then run:

```bash
POLICY_GPU=0 bash tests/pi_05/test_libero_sft_alignment.sh
```

The runner forces Hugging Face offline mode and does not download assets.

## 2026-09-19 Gate 4 result: PASS

The real local `lerobot/libero` sample at index 0 yielded an `(10, 7)` action
chunk and `(10,)` padding mask. The model received identical train/inference
images, masks, normalized state, token IDs and token mask at exact tensor
equality. Quantile-normalized 7D actions matched an independent calculation
within `1e-6`; the remaining 25 dimensions were exactly zero. A real
`pi05_libero_pytorch` forward produced finite loss `0.000174077039` in the
final run. Noise and time are sampled, so this loss value is not a fixed
regression target. The test used a `forkserver` DataLoader worker, matching
the trainer's multiprocessing boundary. Result: `2 passed in 69.79s`, exit 0.

Additional CPU regressions: PI0.5 modeling/inference and the new unit test
returned `26 passed, 1 skipped` (the real CUDA case skipped without assets);
SFT config/trainer quick tests returned `13 passed, 1 deselected` (the existing
two-step GPU SFT test was not run). No optimizer step, backward pass, checkpoint
write or SFT trial was performed. Gate 5 remains separate.

Provenance: local Carrot HEAD `d3ac7484494088af0a6759283a62ec05b038d14c`
plus uncommitted Gate 4 files synced to remote. Remote source has no `.git`;
`src/carrot/data/libero.py` SHA-256 is
`e9c55172e1b1c373de322774217084633d74adee2bca5d9b5c00995a3ec1ff7d`,
`src/carrot/models/pi05/loss_fn.py` SHA-256 is
`59445d246d87b6b2400a602f9d13f5d150cb55fadcd69abbb335836b20b7c395`,
and `src/carrot/models/pi05/transforms.py` SHA-256 is
`7cb78908fa6f3e02f19aa46932f45a3a45d0fc06105d024949b35faea8bc6fa3`.
The checkpoint is `${MY_DFS}/hf-hub/Physical-Intelligence/pi05_libero_pytorch`
(`model.safetensors` 7,233,650,408 bytes; config SHA-256
`5c2728c53f4b33ee16380140f303713fdb5df78a8d26ee1489ace374fc54327a`),
stats SHA-256 `b3a44bb2810436fb62917decaea58bd4d9110255df527dea21e8fd40c960bd84`,
and dataset `${MY_DFS}/hf-hub/lerobot/libero` (`meta/info.json` SHA-256
`630630a1a4f07fc9fc1f1b2b7987ed3bd034bd8cffaaf8a0e6939ae9c468445d`).
The container was `mpi-launcher@mpi-1759754893-launcher`; its image tag was
not exposed, so it is not attributed to a guessed tag.

Actual command: `MY_DFS=<detected personal DFS> POLICY_GPU=0 bash
tests/pi_05/test_libero_sft_alignment.sh`. The runner activates
`/opt/venvs/carrot`, exports source `PYTHONPATH`, and forces offline loading.
Ray was started per test environment convention and stopped after testing;
dguard was restored to `DGUARD_WATCH=1` with no scheduled restore.

## 2026-09-26 stats-source regression: BLOCKED

The new CPU regression checks configured-file precedence and failure on a
missing selected stats file. The existing runner still executes the whole
`test_libero_sft_alignment.py` file; no runner change is needed.

Local command: `PYTHONPATH=src:tests python -m pytest -q tests/pi_05/test_libero_sft_alignment.py`.
The command used `/home/nrwu/programs/miniconda2/envs/py312/bin/python`
(Python 3.12.13) from Carrot HEAD `31d485c` plus the working-tree changes;
no Docker image was involved. It stopped during collection with exit code 2
because this environment lacks `lerobot`
(`ModuleNotFoundError: No module named 'lerobot'`).
No test case ran; the historical Gate 4 PASS above is unchanged. The current
changes passed `ruff check`, `py_compile`, and `git diff --check` locally.
