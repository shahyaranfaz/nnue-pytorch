import pytest
import torch

from model.config import LossParams
from model.lambda_utils import LambdaController


def _actual_lambda(loss_params):
    return LambdaController()(
        loss_params,
        current_epoch=5,
        max_epoch=10,
        is_training=True,
        scorenet=torch.zeros(1),
    )


def test_lambda_shorthand_remains_the_effective_constant():
    loss_params = LossParams(lambda_=0.74)

    assert loss_params.start_lambda is None
    assert loss_params.end_lambda is None
    assert _actual_lambda(loss_params) == pytest.approx(0.74)


def test_explicit_lambda_range_overrides_shorthand():
    loss_params = LossParams(
        lambda_=0.74,
        start_lambda=0.5,
        end_lambda=0.9,
    )

    assert _actual_lambda(loss_params) == pytest.approx(0.7)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"start_lambda": 0.5},
        {"end_lambda": 0.5},
    ],
)
def test_lambda_range_requires_both_endpoints(kwargs):
    with pytest.raises(ValueError):
        LossParams(**kwargs)
