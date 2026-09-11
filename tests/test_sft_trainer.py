import torch
from torch import nn
from torch.utils.data import DataLoader

from carrot.trainer.sft.config import SFTConfig
from carrot.trainer.sft.trainer import SFTTrainer, _scheduler


class FakePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(1.0))

    def forward(self, batch):
        loss = ((self.weight * batch - 0.0) ** 2).mean()
        return loss, {}


def test_sft_trainer_accumulates_gradients() -> None:
    model = FakePolicy()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    config = SFTConfig.from_dict(
        {
            "steps": 2,
            "gradient_accumulation_steps": 2,
            "save_freq": 0,
            "log_freq": 10,
            "fsdp": {"enabled": False},
        }
    )
    trainer = SFTTrainer(
        model=model,
        optimizer=optimizer,
        scheduler=_scheduler(optimizer, 0, config.steps),
        preprocessor=lambda batch: batch,
        dataloader=DataLoader([torch.tensor([1.0])] * 4, batch_size=1),
        config=config,
    )

    metrics = trainer.train()

    assert metrics["step"] == 2
    assert model.weight.item() < 1.0
