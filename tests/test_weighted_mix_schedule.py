import pytest

from data_loader.mix_schedule import secondary_batch, source_batches_before


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
