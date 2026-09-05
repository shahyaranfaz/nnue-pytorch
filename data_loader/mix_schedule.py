def secondary_batch(batch_index: int, secondary_batches: int, cycle_batches: int) -> bool:
    """Choose a source deterministically while distributing secondary batches evenly."""
    if batch_index < 0:
        raise ValueError("batch_index must be non-negative")
    if not 0 < secondary_batches < cycle_batches:
        raise ValueError("secondary_batches must be between 1 and cycle_batches - 1")
    current = (batch_index * secondary_batches) // cycle_batches
    following = ((batch_index + 1) * secondary_batches) // cycle_batches
    return following != current


def source_batches_before(batch_index: int, secondary_batches: int, cycle_batches: int):
    secondary = (batch_index * secondary_batches) // cycle_batches
    return batch_index - secondary, secondary
