import gc
import hashlib
import json
import os
import random
import shutil
import time
from pathlib import Path
from typing import override

import numpy as np
import torch
from transformers import AutoTokenizer

from carrot.distributed.worker import Worker
from carrot.models.pi05.embodiments.libero import create_libero_transform_spec
from carrot.models.pi05.model import PI0Pytorch
from carrot.trainer.rl.algorithms import chunk_logprob, duration_gae, ppo_loss
from carrot.trainer.rl.config import TransferConfig
from carrot.trainer.rl.data import Demonstrations
from carrot.trainer.rl.pi05 import FlowPolicy


class LearnerWorker(Worker):
    def __init__(self, config: TransferConfig) -> None:
        self.config = config
        self.policy: FlowPolicy | None = None
        self.pending: dict | None = None
        self.buffer: list[dict] = []
        self.transitions = self.updates = self.sft_updates = 0
        self.version = 0
        self.sft_checkpoint = str(Path(config.output) / "checkpoints/sft")

    @override
    def setup(self) -> None:
        assert torch.cuda.is_available()
        assert self.master_addr == self.config.learner_node_ip
        torch.cuda.set_device(self.local_rank)
        self.device = torch.device("cuda", self.local_rank)
        assert torch.cuda.get_device_properties(self.device).total_memory >= 48 * 1024**3
        torch.manual_seed(self.config.seed)
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.tokenizer, fix_mistral_regex=True, local_files_only=True
        )
        self.norm_hash = hashlib.sha256(Path(self.config.norm_stats).read_bytes()).hexdigest()
        document = json.loads(Path(self.config.norm_stats).read_text())
        self.stats = document["norm_stats"]
        self.load(self.config.checkpoint)

    def load(self, checkpoint: str, embodiment: str = "paint") -> dict:
        """Replace policy and optimizer, retaining the experiment's fixed normalization.

        Parameters
        ----------
        checkpoint : str
            Local OpenPI export; this does not restore learner state or counters.
        embodiment : str
            Task label; both tasks retain the original LIBERO camera masks and transforms.

        Returns
        -------
        dict
            Model, device and normalization provenance.
        """
        assert self.pending is None and not self.buffer
        assert embodiment in ("paint", "libero")
        if self.policy is not None:
            del self.optimizer
            self.policy = None
            gc.collect()
            torch.cuda.empty_cache()
        model = PI0Pytorch.from_pretrained(checkpoint)
        spec = create_libero_transform_spec(
            self.tokenizer, self.stats, model_action_dim=model.config.action_dim
        )
        self.policy = FlowPolicy(
            model,
            spec,
            device=str(self.device),
            num_steps=self.config.denoise_steps,
            noise_level=self.config.noise_level,
        )
        self.optimizer = torch.optim.AdamW(
            [
                {"params": self.policy.parameters(), "lr": self.config.policy_lr},
                {"params": self.policy.value_head.parameters(), "lr": self.config.value_lr},
            ],
            weight_decay=0,
        )
        return {
            "checkpoint": checkpoint,
            "embodiment": embodiment,
            "device": torch.cuda.get_device_name(self.device),
            "norm_sha256": self.norm_hash,
        }

    def infer(self, observations: list[dict], seed: int | None = None) -> np.ndarray:
        assert self.policy is not None
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        return self.policy.sample(observations, stochastic=False)["actions"]

    def sample(self, observations: list[dict]) -> np.ndarray:
        """Sample one on-policy chunk and retain its replay data until ``record``.

        Parameters
        ----------
        observations : list[dict]

        Returns
        -------
        numpy.ndarray
            Decoded actions with shape (env, 10, 7); only one pending sample is allowed.
        """
        assert self.policy is not None and self.pending is None
        batches = [
            self.policy.sample(observations[i : i + self.config.micro_batch_size])
            for i in range(0, len(observations), self.config.micro_batch_size)
        ]
        sample = {
            key: (
                np.concatenate([b[key] for b in batches])
                if key == "actions"
                else torch.cat([b[key] for b in batches])
            )
            for key in batches[0]
        }
        self.pending = {**sample, "observations": observations, "version": self.version}
        return self.pending["actions"]

    def record(self, result: dict) -> dict:
        """Record actual chunk durations and value the final observations before reset.

        Parameters
        ----------
        result : dict
            Simulator step result corresponding to the current pending sample.

        Returns
        -------
        dict
            Actual control-transition count, buffered chunks and mean reward.
        """
        assert self.policy is not None and self.pending is not None
        assert self.pending["version"] == self.version
        assert len(result["observations"]) == len(self.pending["observations"])
        entry = {
            **self.pending,
            **{
                k: torch.tensor(result[k])
                for k in ("rewards", "lengths", "terminated", "truncated")
            },
            "next_values": self.policy.values(result["observations"]),
        }
        del entry["actions"]
        self.transitions += int(entry["lengths"].sum())
        self.buffer.append(entry)
        self.pending = None
        return {
            "transitions": self.transitions,
            "buffer_chunks": len(self.buffer),
            "reward_mean": float(entry["rewards"].mean()),
        }

    def update(self) -> dict:
        """Perform one on-policy PPO epoch over the collected SMDP chunks.

        Returns
        -------
        dict
            Loss, ratio parity, gradients, clipping, KL, actual transitions and peak GPU memory.
        """
        assert self.policy is not None and self.pending is None and self.buffer
        cfg = self.config
        fields = ("rewards", "value", "next_values", "lengths", "terminated", "truncated")
        stacked = {k: torch.stack([entry[k] for entry in self.buffer]) for k in fields}
        advantages, returns = duration_gae(
            stacked["rewards"],
            stacked["value"],
            stacked["next_values"],
            stacked["lengths"],
            stacked["terminated"],
            stacked["truncated"],
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
        )
        advantages = advantages.flatten()
        advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)
        returns = returns.flatten()
        raw = [obs for entry in self.buffer for obs in entry["observations"]]
        replay = {
            k: torch.cat([entry[k] for entry in self.buffer])
            for k in ("before", "after", "index", "logprob", "lengths", "value")
        }
        old = chunk_logprob(replay["logprob"], replay["lengths"])
        max_parity = 0.0
        # Check every stored transition before any optimizer step, including uneven chunk durations.
        with torch.no_grad():
            for start in range(0, len(raw), cfg.micro_batch_size):
                end = start + cfg.micro_batch_size
                actual, _ = self.policy.evaluate(
                    raw[start:end], {k: v[start:end] for k, v in replay.items()}
                )
                actual = chunk_logprob(actual, replay["lengths"][start:end].to("cuda"))
                max_parity = max(max_parity, float((actual.cpu() - old[start:end]).abs().max()))
        assert max_parity < 0.02, f"pre-update logprob mismatch: {max_parity}"
        order = torch.randperm(len(raw))
        metrics: dict[str, list[float]] = {}
        for start in range(0, len(raw), cfg.minibatch_size):
            batch = order[start : start + cfg.minibatch_size]
            self.optimizer.zero_grad(set_to_none=True)
            for offset in range(0, len(batch), cfg.micro_batch_size):
                ids = batch[offset : offset + cfg.micro_batch_size]
                logprob, values = self.policy.evaluate(
                    [raw[i] for i in ids.tolist()], {k: v[ids] for k, v in replay.items()}
                )
                logprob = chunk_logprob(logprob, replay["lengths"][ids].to("cuda"))
                loss, stats = ppo_loss(
                    logprob,
                    old[ids].to("cuda"),
                    advantages[ids].to("cuda"),
                    values,
                    replay["value"][ids].to("cuda"),
                    returns[ids].to("cuda"),
                    cfg.clip,
                )
                assert torch.isfinite(loss)
                (loss * len(ids) / len(batch)).backward()
                for key, value in stats.items():
                    metrics.setdefault(key, []).append(float(value))
            parameters = self.policy.parameters() + list(self.policy.value_head.parameters())
            norm = torch.nn.utils.clip_grad_norm_(parameters, 1.0, error_if_nonfinite=True)
            metrics.setdefault("grad_norm", []).append(float(norm))
            self.optimizer.step()
        assert all(
            p.grad is None for p in self.policy.model.paligemma_with_expert.paligemma.parameters()
        )
        self.updates += 1
        self.version += 1
        self.buffer.clear()
        return {
            **{k: float(np.mean(v)) for k, v in metrics.items()},
            "max_preupdate_logprob_error": max_parity,
            "updates": self.updates,
            "transitions": self.transitions,
            "version": self.version,
            "peak_gpu_gib": torch.cuda.max_memory_allocated() / 1024**3,
        }

    def sft(self, directory: str, deadline: float) -> dict:
        """Fit the action expert to successful source demonstrations within a deadline.

        Parameters
        ----------
        directory : str
        deadline : float
            Absolute Unix time; an in-progress update finishes before stopping.

        Returns
        -------
        dict
            Completed SFT updates and dataset transitions; no target data is consumed.
        """
        assert self.policy is not None
        cfg = self.config
        dataset = Demonstrations(directory)
        optimizer = torch.optim.AdamW(self.policy.parameters(), lr=cfg.sft_lr, weight_decay=0)
        log = Path(cfg.output) / "sft.jsonl"
        while self.sft_updates < cfg.sft_steps and time.time() < deadline:
            indices = np.random.randint(len(dataset), size=cfg.sft_batch_size)
            optimizer.zero_grad(set_to_none=True)
            total = 0.0
            for offset in range(0, len(indices), cfg.micro_batch_size):
                batch = [dataset[int(i)] for i in indices[offset : offset + cfg.micro_batch_size]]
                raw, lengths = zip(*batch, strict=True)
                loss = self.policy.supervised_loss(list(raw), torch.tensor(lengths))
                assert torch.isfinite(loss)
                (loss * len(batch) / cfg.sft_batch_size).backward()
                total += float(loss.detach()) * len(batch) / cfg.sft_batch_size
            norm = torch.nn.utils.clip_grad_norm_(
                self.policy.parameters(), 1, error_if_nonfinite=True
            )
            optimizer.step()
            self.sft_updates += 1
            row = {
                "step": self.sft_updates,
                "loss": total,
                "grad_norm": float(norm),
                "time": time.time(),
            }
            with log.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
        del optimizer
        return {"sft_updates": self.sft_updates, "dataset_transitions": len(dataset)}

    def save(self, directory: str) -> dict:
        """Atomically export inference weights and exact FP32 learner state at an update boundary.

        Parameters
        ----------
        directory : str
            Must not exist; environment state is intentionally excluded from resumption.

        Returns
        -------
        dict
            Checkpoint location, counters and normalization provenance.
        """
        assert self.policy is not None and self.pending is None and not self.buffer
        destination = Path(directory)
        temporary = destination.with_name(destination.name + ".partial")
        assert not destination.exists() and not temporary.exists()
        temporary.mkdir(parents=True)
        self.policy.model.save_pretrained(temporary)
        assets = temporary / "assets/physical-intelligence/libero"
        assets.mkdir(parents=True)
        shutil.copyfile(self.config.norm_stats, assets / "norm_stats.json")
        state = {
            "format": 1,
            "trainable": {
                n: p.detach().cpu()
                for n, p in self.policy.model.named_parameters()
                if p.requires_grad
            },
            "value_head": self.policy.value_head.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all(),
            "numpy_rng": np.random.get_state(),
            "python_rng": random.getstate(),
            "transitions": self.transitions,
            "updates": self.updates,
            "sft_updates": self.sft_updates,
            "version": self.version,
            "norm_sha256": self.norm_hash,
            "denoise_steps": self.config.denoise_steps,
            "noise_level": self.config.noise_level,
        }
        torch.save(state, temporary / "learner.pt")
        manifest = {
            "transitions": self.transitions,
            "updates": self.updates,
            "sft_updates": self.sft_updates,
            "norm_sha256": self.norm_hash,
            "resume": "exact learner state; fresh simulator episodes",
            "initial_checkpoint": self.config.checkpoint,
            "sft_checkpoint": self.sft_checkpoint,
        }
        (temporary / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        os.replace(temporary, destination)
        return {"checkpoint": str(destination), **manifest}

    def resume(self, directory: str) -> dict:
        """Restore trusted FP32 learner state on the original frozen initialization.

        Parameters
        ----------
        directory : str
            Local checkpoint with matching normalization, SDE settings and initialization.

        Returns
        -------
        dict
            Restored counters; the caller must restart simulator episodes.
        """
        assert self.policy is not None and not self.buffer and self.pending is None
        manifest = json.loads((Path(directory) / "manifest.json").read_text())
        assert manifest["initial_checkpoint"] == self.config.checkpoint
        self.sft_checkpoint = manifest["sft_checkpoint"]
        state = torch.load(Path(directory) / "learner.pt", map_location="cpu", weights_only=False)
        assert state["format"] == 1 and state["norm_sha256"] == self.norm_hash
        assert state["denoise_steps"] == self.config.denoise_steps
        assert state["noise_level"] == self.config.noise_level
        trainable = {n: p for n, p in self.policy.model.named_parameters() if p.requires_grad}
        assert set(trainable) == set(state["trainable"])
        with torch.no_grad():
            for name, param in trainable.items():
                assert param.dtype == torch.float32
                param.copy_(state["trainable"][name])
        self.policy.value_head.load_state_dict(state["value_head"], strict=True)
        self.optimizer.load_state_dict(state["optimizer"])
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])
        np.random.set_state(state["numpy_rng"])
        random.setstate(state["python_rng"])
        self.transitions, self.updates = state["transitions"], state["updates"]
        self.sft_updates, self.version = state["sft_updates"], state["version"]
        return {
            "transitions": self.transitions,
            "updates": self.updates,
            "fresh_simulator_episodes_required": True,
        }
