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


def weighted_cycle(source_batches: tuple[int, ...]) -> tuple[int, ...]:
    """Build a deterministic, evenly distributed weighted source cycle."""
    if len(source_batches) < 2 or any(weight <= 0 for weight in source_batches):
        raise ValueError("source_batches must contain at least two positive weights")

    cycle_batches = sum(source_batches)
    used = [0] * len(source_batches)
    cycle = []

    for batch_index in range(cycle_batches):
        source = max(
            range(len(source_batches)),
            key=lambda index: (
                (batch_index + 1) * source_batches[index]
                - used[index] * cycle_batches,
                -index,
            ),
        )
        cycle.append(source)
        used[source] += 1

    if tuple(used) != source_batches:
        raise RuntimeError("weighted cycle did not preserve source weights")
    return tuple(cycle)


def weighted_source(batch_index: int, source_batches: tuple[int, ...]) -> int:
    if batch_index < 0:
        raise ValueError("batch_index must be non-negative")
    cycle = weighted_cycle(source_batches)
    return cycle[batch_index % len(cycle)]


def weighted_source_batches_before(
    batch_index: int, source_batches: tuple[int, ...]
) -> tuple[int, ...]:
    if batch_index < 0:
        raise ValueError("batch_index must be non-negative")

    cycle = weighted_cycle(source_batches)
    complete_cycles, remainder = divmod(batch_index, len(cycle))
    counts = [complete_cycles * weight for weight in source_batches]
    for source in cycle[:remainder]:
        counts[source] += 1
    return tuple(counts)
