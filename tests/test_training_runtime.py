"""Functional regressions for updates, checkpoint loading and distributed metrics."""
import copy
import datetime
import time

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, TensorDataset

from kmerformer import train, train_ddp
from kmerformer.evaluate import resolve_num_classes
from kmerformer.training_utils import checkpoint_weights, optimizer_step


class TinyClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = nn.Linear(2, 3)
        self.head = nn.Linear(3, 2)
        self.register_buffer("offset", torch.zeros(2))

    def forward(self, inputs, mask):
        return self.head(self.backbone(inputs)) + self.offset

    def get_backbone_params(self):
        return list(self.backbone.parameters())

    def get_head_params(self):
        return list(self.head.parameters())


def train_one(module, model, loader, optimizer, scheduler, accumulation):
    return module.train_epoch(
        model, loader, nn.CrossEntropyLoss(), optimizer, scheduler,
        torch.amp.GradScaler("cuda", enabled=False), torch.device("cpu"),
        accumulation, 100.0, False, amp_dtype=torch.float16)


@pytest.mark.parametrize("module", [train, train_ddp])
@pytest.mark.parametrize("count,accumulation", [(3, 4), (5, 2)])
def test_accumulation_matches_actual_group_means(module, count, accumulation):
    torch.manual_seed(123)
    model = TinyClassifier()
    reference = copy.deepcopy(model)
    x = torch.randn(count, 2)
    y = torch.arange(count) % 2
    loader = DataLoader(TensorDataset(x, torch.ones_like(x), y), batch_size=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)
    ref_scheduler = torch.optim.lr_scheduler.ExponentialLR(ref_optimizer, gamma=0.7)
    train_one(module, model, loader, optimizer, scheduler, accumulation)
    for start in range(0, count, accumulation):
        stop = start + accumulation
        ref_optimizer.zero_grad()
        nn.functional.cross_entropy(reference(x[start:stop], None), y[start:stop]).backward()
        ref_optimizer.step()
        ref_scheduler.step()
    for got, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(got, expected)
    assert scheduler.last_epoch == ref_scheduler.last_epoch == (count + accumulation - 1) // accumulation


@pytest.mark.parametrize("module", [train, train_ddp])
def test_invalid_loss_discards_pending_group_then_recovers(module):
    torch.manual_seed(456)
    model = TinyClassifier()
    reference = copy.deepcopy(model)
    x = torch.tensor([[1., 2.], [float("nan"), 1.], [2., 1.]])
    y = torch.tensor([0, 1, 0])
    loader = DataLoader(TensorDataset(x, torch.ones_like(x), y), batch_size=1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    train_one(module, model, loader, optimizer, None, 4)
    ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
    nn.functional.cross_entropy(reference(x[2:], None), y[2:]).backward()
    ref_optimizer.step()
    for got, expected in zip(model.parameters(), reference.parameters()):
        torch.testing.assert_close(got, expected)


def test_sparse_overflow_skips_both_optimizers_and_recovers():
    embedding = nn.Embedding(8, 2, sparse=True)
    head = nn.Linear(2, 2)
    parameters = list(embedding.parameters()) + list(head.parameters())
    opts = [torch.optim.SparseAdam(embedding.parameters(), lr=0.1),
            torch.optim.AdamW(head.parameters(), lr=0.1)]
    scheds = [torch.optim.lr_scheduler.ExponentialLR(o, gamma=0.9) for o in opts]
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    before = [p.detach().clone() for p in parameters]
    head(embedding(torch.tensor([1, 2]))).sum().backward()
    embedding.weight.grad._values()[0, 0] = float("inf")
    assert not optimizer_step(opts, scheds, scaler, parameters, 1.0)
    for got, expected in zip(parameters, before):
        torch.testing.assert_close(got, expected)
    assert all(s.last_epoch == 0 for s in scheds)
    head(embedding(torch.tensor([1, 2]))).sum().backward()
    assert optimizer_step(opts, scheds, scaler, parameters, 1.0)
    assert all(s.last_epoch == 1 for s in scheds)
    assert all(o.state for o in opts)


def test_checkpoint_mapping_preserves_native_and_adapts_legacy_keys():
    native = nn.Module()
    native.query = nn.Linear(2, 2)
    state = {k: torch.full_like(v, 0.25) for k, v in native.state_dict().items()}
    native.load_state_dict(checkpoint_weights(native, state))
    torch.testing.assert_close(native.query.weight, state["query.weight"])
    adapted = nn.Module()
    adapted.query = nn.Module()
    adapted.query.base_layer = nn.Linear(2, 2)
    adapted.load_state_dict(checkpoint_weights(adapted, state))
    torch.testing.assert_close(adapted.query.base_layer.weight, state["query.weight"])
    with pytest.raises(RuntimeError, match="Missing key"):
        native.load_state_dict(checkpoint_weights(native, {"unrelated.weight": torch.ones(1)}))


def test_ddp_head_lr_and_legacy_optimizer_resume():
    model = TinyClassifier()
    groups = train_ddp.optimizer_groups(model, 0.001, 0.01)
    optimizer = torch.optim.AdamW(groups)
    assert {id(p) for p in optimizer.param_groups[1]["params"]} == {id(p) for p in model.head.parameters()}
    assert optimizer.param_groups[1]["lr"] == 0.01
    legacy = torch.optim.AdamW([{"params": list(model.parameters()), "lr": 0.001},
                               {"params": [], "lr": 0.01}])
    model(torch.ones(1, 2), None).sum().backward()
    legacy.step()
    state = legacy.state_dict()
    restored = torch.optim.AdamW(train_ddp.optimizer_groups(model, 1., 2., state))
    restored.load_state_dict(state)
    assert restored.param_groups[0]["lr"] == 0.001
    assert restored.param_groups[1]["params"] == []
    for parameter in model.parameters():
        torch.testing.assert_close(restored.state[parameter]["exp_avg"], legacy.state[parameter]["exp_avg"])


@pytest.mark.parametrize("labels,detected,override,expected", [([7, 119], 2, 120, 120),
                                                            ([1, 2], 2, 3, 3), ([0, 1], 2, None, 2)])
def test_evaluation_uses_head_size_without_renumbering(labels, detected, override, expected):
    assert resolve_num_classes(labels, detected, override) == expected


@pytest.mark.parametrize("labels,override", [([7, 119], 2), ([-1, 0], 2), ([], 2), ([0], 0)])
def test_invalid_evaluation_labels_fail_early(labels, override):
    with pytest.raises(ValueError):
        resolve_num_classes(labels, 2, override)


def _distributed_worker(rank, rendezvous):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2,
                            timeout=datetime.timedelta(seconds=15))
    try:
        model = TinyClassifier()
        with torch.no_grad():
            for p in model.parameters():
                p.zero_()
            model.head.bias[0] = 1.
        model = DDP(model)
        criterion = nn.CrossEntropyLoss()
        for count in (5, 1):
            x = torch.ones(count, 2)
            y = torch.zeros(count, dtype=torch.long)
            y[0] = 1
            data = TensorDataset(x, x, y)
            loader = DataLoader(data, batch_size=2,
                                sampler=train_ddp.DistributedEvalSampler(data, 2, rank))
            loss, accuracy = train_ddp.validate(model, loader, criterion, torch.device("cpu"),
                                                False, torch.float16, 2)
            assert accuracy == pytest.approx((count - 1) / count)
            expected = criterion(model.module(x, x), y).item()
            assert loss == pytest.approx(expected)

        # Only rank zero receives a NaN. Both ranks must discard the group and
        # finish the next update without hanging or diverging.
        x = torch.ones(3, 2)
        if rank == 0:
            x[0, 0] = float("nan")
        data = TensorDataset(x, torch.ones_like(x), torch.ones(3, dtype=torch.long))
        loader = DataLoader(data, batch_size=1)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        train_one(train_ddp, model, loader, optimizer, None, 4)
        weights = torch.cat([p.detach().flatten() for p in model.parameters()])
        gathered = [torch.empty_like(weights) for _ in range(2)]
        dist.all_gather(gathered, weights)
        torch.testing.assert_close(gathered[0], gathered[1])
        assert torch.isfinite(weights).all()

        saved = []
        result = train_ddp.train_epoch(
            model, loader, criterion, optimizer, None,
            torch.amp.GradScaler("cuda", enabled=False), torch.device("cpu"),
            1, 1., False, torch.float16, is_main=(rank == 0),
            time_limit_sec=5, job_start=time.time() - (10 if rank == 0 else 0),
            periodic_save_fn=saved.append)
        assert result[2] and saved == [0]
    finally:
        dist.destroy_process_group()


def test_two_rank_validation_nan_recovery_and_synchronized_timeout(tmp_path):
    context = mp.spawn(_distributed_worker,
                       args=((tmp_path / "rendezvous").as_uri(),), nprocs=2, join=False)
    deadline = time.monotonic() + 50
    try:
        while not context.join(timeout=1):
            if time.monotonic() > deadline:
                pytest.fail("Distributed regression exceeded 50 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=3)
