import struct
from io import BytesIO

import torch

from model.shayveri_model import ShayveriBucketedModel, ShayveriDirectModel
from model.lightning_module import calculate_bullet_loss
from model.utils.shayveri_serialize import ShayveriNNUEReader, ShayveriNNUEWriter


def test_direct_model_selects_stm_perspective():
    model = ShayveriDirectModel()
    with torch.no_grad():
        model.input.weight.zero_()
        model.input.virtual_weight.zero_()
        model.input.bias.zero_()
        model.input.weight[10, 0] = 0.5
        model.input.weight[20, 0] = 0.25
        model.output.weight.zero_()
        model.output.bias.zero_()
        model.output.weight[0, 0] = 1.0
        model.output.weight[0, 512] = 2.0

    white_indices = torch.tensor([[10, -1]], dtype=torch.int32)
    black_indices = torch.tensor([[20, -1]], dtype=torch.int32)
    piece_count = torch.tensor([2], dtype=torch.int32)

    white = model(
        torch.ones(1, 1),
        torch.zeros(1, 1),
        white_indices,
        black_indices,
        piece_count,
        False,
        False,
    )
    black = model(
        torch.zeros(1, 1),
        torch.ones(1, 1),
        white_indices,
        black_indices,
        piece_count,
        False,
        False,
    )

    assert torch.allclose(white, torch.tensor([[0.375]]))
    assert torch.allclose(black, torch.tensor([[0.5625]]))


def test_legacy_pickled_direct_model_gains_post_net1_defaults():
    model = ShayveriDirectModel(use_factorizer=True)
    del model.output_factorizer
    del model.output_buckets
    del model.num_ls_buckets

    checkpoint = BytesIO()
    torch.save(model, checkpoint)
    checkpoint.seek(0)
    loaded = torch.load(checkpoint, weights_only=False)

    assert loaded.output_buckets == 1
    assert loaded.num_ls_buckets == 1
    assert loaded.output_factorizer is None
    assert loaded.merged_output_weight() is loaded.output.weight
    assert loaded.merged_output_bias() is loaded.output.bias


def test_bullet_loss_matches_blended_sigmoid_mse():
    prediction = torch.tensor([[0.0], [400.0]])
    score = torch.tensor([[400.0], [-400.0]])
    outcome = torch.tensor([[1.0], [0.0]])
    target = 0.3 * outcome + 0.7 * torch.sigmoid(score / 400.0)
    expected = ((torch.sigmoid(prediction / 400.0) - target) ** 2).mean()
    assert torch.allclose(
        calculate_bullet_loss(prediction, score, outcome, 0.3, 400.0),
        expected,
    )


def test_writer_matches_shayveri_layout_and_integer_evaluation():
    model = ShayveriDirectModel()
    with torch.no_grad():
        model.input.weight.zero_()
        model.input.virtual_weight.zero_()
        model.input.bias.zero_()
        model.input.weight[10, 0] = 100 / 255
        model.input.weight[20, 0] = 50 / 255
        model.output.weight.zero_()
        model.output.bias.fill_(1234 / (255 * 255))
        model.output.weight[0, 0] = 7 / 255
        model.output.weight[0, 512] = -3 / 255

    buf = ShayveriNNUEWriter(model).buf
    expected_size = 12 + 12288 * 512 * 2 + 512 * 2 + 1024 * 2 + 4
    assert len(buf) == expected_size
    assert struct.unpack_from("<III", buf) == (0x4E4E5545, 3, 16)

    weights_offset = 12
    bias_offset = weights_offset + 12288 * 512 * 2
    output_offset = bias_offset + 512 * 2
    output_bias_offset = output_offset + 1024 * 2
    assert struct.unpack_from("<h", buf, weights_offset + (10 * 512) * 2)[0] == 100
    assert struct.unpack_from("<h", buf, weights_offset + (20 * 512) * 2)[0] == 50
    assert struct.unpack_from("<h", buf, output_offset)[0] == 7
    assert struct.unpack_from("<h", buf, output_offset + 512 * 2)[0] == -3
    assert struct.unpack_from("<i", buf, output_bias_offset)[0] == 1234

    engine_sum = ((100 * 100) // 255) * 7 + ((50 * 50) // 255) * -3 + 1234
    engine_cp = int(engine_sum * 400 / (255 * 255))
    assert engine_cp == 9

    output = model(
        torch.ones(1, 1),
        torch.zeros(1, 1),
        torch.tensor([[10, -1]], dtype=torch.int32),
        torch.tensor([[20, -1]], dtype=torch.int32),
        torch.tensor([2], dtype=torch.int32),
        True,
        True,
    )
    assert torch.allclose(output, torch.tensor([[engine_sum / (255 * 255)]]))


def test_reader_round_trip_is_exact_and_enables_zeroed_factorizer():
    model = ShayveriDirectModel()
    with torch.no_grad():
        model.input.weight.uniform_(-0.5, 0.5)
        model.input.bias.uniform_(-0.5, 0.5)
        model.output.weight.uniform_(-0.5, 0.5)
        model.output.bias.uniform_(-0.5, 0.5)

    original = ShayveriNNUEWriter(model).buf
    loaded = ShayveriNNUEReader(
        BytesIO(original),
        use_factorizer=True,
    ).model

    assert loaded.input.virtual_weight.requires_grad
    assert torch.count_nonzero(loaded.input.virtual_weight) == 0
    assert ShayveriNNUEWriter(loaded).buf == original


def test_bucketed_model_selects_material_head():
    model = ShayveriBucketedModel()
    with torch.no_grad():
        model.input.weight.zero_()
        model.input.virtual_weight.zero_()
        model.input.bias.zero_()
        model.output.weight.zero_()
        model.output.bias.copy_(torch.arange(8, dtype=torch.float32))

    piece_count = torch.tensor(
        [1, 4, 5, 8, 9, 12, 13, 16, 17, 20, 21, 24, 25, 28, 29, 32],
        dtype=torch.int64,
    )
    batch_size = piece_count.numel()
    output = model(
        torch.ones(batch_size, 1),
        torch.zeros(batch_size, 1),
        torch.full((batch_size, 1), -1, dtype=torch.int32),
        torch.full((batch_size, 1), -1, dtype=torch.int32),
        piece_count,
        False,
        False,
    )
    assert torch.equal(output.reshape(-1), torch.arange(8).repeat_interleave(2))


def test_v3_warm_expansion_is_function_preserving():
    parent = ShayveriDirectModel()
    with torch.no_grad():
        parent.input.weight.uniform_(-0.1, 0.1)
        parent.input.bias.uniform_(-0.1, 0.1)
        parent.output.weight.uniform_(-0.1, 0.1)
        parent.output.bias.uniform_(-0.1, 0.1)

    v3 = ShayveriNNUEWriter(parent).buf
    reference = ShayveriNNUEReader(BytesIO(v3), output_buckets=1).model
    child = ShayveriNNUEReader(
        BytesIO(v3), use_factorizer=True, output_buckets=8
    ).model
    assert isinstance(child, ShayveriBucketedModel)
    assert child.output_factorizer is not None
    assert torch.count_nonzero(child.output_factorizer.weight) == 0
    assert torch.count_nonzero(child.output_factorizer.bias) == 0
    for bucket in range(8):
        assert torch.equal(child.output.weight[bucket], child.output.weight[0])
        assert torch.equal(child.output.bias[bucket], child.output.bias[0])

    batch_size = 32
    white_indices = torch.randint(0, 12288, (batch_size, 24), dtype=torch.int32)
    black_indices = torch.randint(0, 12288, (batch_size, 24), dtype=torch.int32)
    piece_count = torch.arange(1, 33, dtype=torch.int64)
    us = torch.randint(0, 2, (batch_size, 1)).float()
    them = 1.0 - us
    reference_output = reference(
        us, them, white_indices, black_indices, piece_count, True, True
    )
    child_output = child(
        us, them, white_indices, black_indices, piece_count, True, True
    )
    torch.testing.assert_close(child_output, reference_output)


def test_bucketed_output_factorizer_updates_shared_and_bucket_parameters():
    model = ShayveriBucketedModel(use_factorizer=True)
    assert model.output_factorizer is not None

    batch_size = 8
    output = model(
        torch.ones(batch_size, 1),
        torch.zeros(batch_size, 1),
        torch.randint(0, 12288, (batch_size, 24), dtype=torch.int32),
        torch.randint(0, 12288, (batch_size, 24), dtype=torch.int32),
        torch.arange(1, 33, 4, dtype=torch.int64),
        False,
        False,
    )
    output.sum().backward()

    assert torch.count_nonzero(model.output_factorizer.weight.grad) > 0
    assert torch.count_nonzero(model.output_factorizer.bias.grad) > 0
    for bucket in range(8):
        assert torch.count_nonzero(model.output.weight.grad[bucket]) > 0
        assert model.output.bias.grad[bucket] != 0


def test_bucketed_output_factorizer_is_folded_during_serialization():
    model = ShayveriBucketedModel(use_factorizer=True)
    assert model.output_factorizer is not None
    with torch.no_grad():
        model.input.weight.zero_()
        model.input.virtual_weight.zero_()
        model.input.bias.zero_()
        model.output.weight.zero_()
        model.output.bias.zero_()
        model.output_factorizer.weight.fill_(7 / 255)
        model.output_factorizer.bias.fill_(1234 / (255 * 255))

    serialized = ShayveriNNUEWriter(model).buf
    loaded = ShayveriNNUEReader(BytesIO(serialized)).model
    for bucket in range(8):
        torch.testing.assert_close(
            loaded.output.weight[bucket],
            torch.full((1024,), 7 / 255),
        )
        torch.testing.assert_close(
            loaded.output.bias[bucket],
            torch.tensor(1234 / (255 * 255)),
        )


def test_v4_bucketed_round_trip_and_section_lengths():
    model = ShayveriBucketedModel(use_factorizer=True)
    with torch.no_grad():
        model.input.weight.uniform_(-0.1, 0.1)
        model.input.bias.uniform_(-0.1, 0.1)
        model.output.weight.uniform_(-0.1, 0.1)
        model.output.bias.uniform_(-0.1, 0.1)

    original = ShayveriNNUEWriter(model).buf
    header = struct.unpack_from("<10I", original)
    assert header[:6] == (0x4E4E5545, 4, 16, 512, 8, 1)
    assert header[6:] == (
        12288 * 512 * 2,
        512 * 2,
        8 * 1024 * 2,
        8 * 4,
    )
    assert len(original) == 40 + sum(header[6:])

    loaded = ShayveriNNUEReader(
        BytesIO(original), use_factorizer=True
    ).model
    assert isinstance(loaded, ShayveriBucketedModel)
    assert loaded.input.virtual_weight.requires_grad
    assert torch.count_nonzero(loaded.input.virtual_weight) == 0
    assert loaded.output_factorizer is not None
    assert torch.count_nonzero(loaded.output_factorizer.weight) == 0
    assert torch.count_nonzero(loaded.output_factorizer.bias) == 0
    assert ShayveriNNUEWriter(loaded).buf == original
