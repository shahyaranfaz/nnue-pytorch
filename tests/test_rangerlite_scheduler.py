import pytest
import torch

from model.optimizers.rangerlite_wrapper import (
    OneCycleCosineRefinementLR,
    SafeOneCycleLR,
)


def _optimizer():
    parameter = torch.nn.Parameter(torch.zeros(()))
    return torch.optim.SGD([parameter], lr=4.375e-4)


def _piecewise(optimizer, one_cycle_steps=20, refinement_steps=10):
    return OneCycleCosineRefinementLR(
        optimizer,
        max_lr=[4.375e-4],
        one_cycle_steps=one_cycle_steps,
        refinement_steps=refinement_steps,
        refinement_lr=2.0e-5,
        refinement_final_lr=1.0e-6,
        div_factor=25,
        final_div_factor=50,
        pct_start=0.2,
    )


def test_piecewise_prefix_matches_safe_one_cycle_exactly():
    actual_optimizer = _optimizer()
    expected_optimizer = _optimizer()
    actual = _piecewise(actual_optimizer)
    expected = SafeOneCycleLR(
        expected_optimizer,
        max_lr=[4.375e-4],
        total_steps=20,
        div_factor=25,
        final_div_factor=50,
        pct_start=0.2,
        cycle_momentum=False,
    )

    assert actual.get_last_lr() == expected.get_last_lr()
    for _ in range(19):
        actual_optimizer.step()
        expected_optimizer.step()
        actual.step()
        expected.step()
        assert actual.get_last_lr() == expected.get_last_lr()


def test_piecewise_boundary_and_cosine_endpoint():
    optimizer = _optimizer()
    scheduler = _piecewise(optimizer)

    for _ in range(20):
        optimizer.step()
        scheduler.step()

    assert scheduler.last_epoch == 20
    assert scheduler.get_last_lr() == pytest.approx([2.0e-5])

    observed = [scheduler.get_last_lr()[0]]
    for _ in range(9):
        optimizer.step()
        scheduler.step()
        observed.append(scheduler.get_last_lr()[0])

    assert observed[-1] == pytest.approx(1.0e-6)
    assert all(left >= right for left, right in zip(observed, observed[1:]))

    optimizer.step()
    scheduler.step()
    assert scheduler.get_last_lr() == pytest.approx([1.0e-6])


def test_piecewise_state_dict_resume_in_refinement():
    optimizer = _optimizer()
    scheduler = _piecewise(optimizer)
    for _ in range(24):
        optimizer.step()
        scheduler.step()

    optimizer_state = optimizer.state_dict()
    scheduler_state = scheduler.state_dict()

    resumed_optimizer = _optimizer()
    resumed_scheduler = _piecewise(resumed_optimizer)
    resumed_optimizer.load_state_dict(optimizer_state)
    resumed_scheduler.load_state_dict(scheduler_state)

    optimizer.step()
    scheduler.step()
    resumed_optimizer.step()
    resumed_scheduler.step()

    assert resumed_scheduler.last_epoch == scheduler.last_epoch
    assert resumed_scheduler.get_last_lr() == scheduler.get_last_lr()
    assert resumed_optimizer.param_groups[0]["lr"] == optimizer.param_groups[0]["lr"]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"one_cycle_steps": 0},
        {"refinement_steps": 1},
        {"refinement_lr": 0.0},
        {"refinement_final_lr": 3.0e-5},
    ],
)
def test_piecewise_rejects_invalid_configuration(kwargs):
    config = {
        "max_lr": [4.375e-4],
        "one_cycle_steps": 20,
        "refinement_steps": 10,
        "refinement_lr": 2.0e-5,
        "refinement_final_lr": 1.0e-6,
        "div_factor": 25,
        "final_div_factor": 50,
        "pct_start": 0.2,
    }
    config.update(kwargs)
    with pytest.raises(ValueError):
        OneCycleCosineRefinementLR(_optimizer(), **config)
