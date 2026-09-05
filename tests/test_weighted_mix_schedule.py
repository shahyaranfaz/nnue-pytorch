import pytest

from data_loader.mix_schedule import (
    secondary_batch,
    source_batches_before,
    weighted_cycle,
    weighted_source,
    weighted_source_batches_before,
)


def test_exact_ratio_in_every_aligned_cycle():
    for start in range(0, 200, 20):
        choices = [secondary_batch(index, 9, 20) for index in range(start, start + 20)]
        assert sum(choices) == 9


def test_resume_counts_match_schedule():
    for batch_index in range(201):
        primary, secondary = source_batches_before(batch_index, 9, 20)
        choices = [secondary_batch(index, 9, 20) for index in range(batch_index)]
        assert secondary == sum(choices)
        assert primary + secondary == batch_index


@pytest.mark.parametrize("secondary,cycle", [(0, 20), (20, 20), (21, 20)])
def test_invalid_schedule_is_rejected(secondary, cycle):
    with pytest.raises(ValueError):
        secondary_batch(0, secondary, cycle)


def test_three_source_cycle_has_exact_weights():
    cycle = weighted_cycle((9, 7, 4))
    assert len(cycle) == 20
    assert tuple(cycle.count(source) for source in range(3)) == (9, 7, 4)


def test_three_source_ratio_repeats_exactly():
    for start in range(0, 200, 20):
        choices = [
            weighted_source(index, (9, 7, 4))
            for index in range(start, start + 20)
        ]
        assert tuple(choices.count(source) for source in range(3)) == (9, 7, 4)


def test_three_source_resume_counts_match_schedule():
    weights = (9, 7, 4)
    for batch_index in range(201):
        counts = weighted_source_batches_before(batch_index, weights)
        choices = [
            weighted_source(index, weights) for index in range(batch_index)
        ]
        assert counts == tuple(choices.count(source) for source in range(3))
        assert sum(counts) == batch_index


@pytest.mark.parametrize("weights", [(), (20,), (9, 0, 11), (9, -1, 12)])
def test_invalid_multi_source_weights_are_rejected(weights):
    with pytest.raises(ValueError):
        weighted_cycle(weights)
