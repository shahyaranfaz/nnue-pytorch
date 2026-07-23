import struct

import torch

from model.shayveri_model import ShayveriDirectModel
from model.lightning_module import calculate_bullet_loss
from model.utils.shayveri_serialize import ShayveriNNUEWriter


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
