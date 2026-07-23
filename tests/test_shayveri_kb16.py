import chess
import struct

from data_loader import DataloaderSkipConfig, SparseBatchProvider
from data_loader import stream
from model.modules.features.shayveri_kb16 import shayveri_kb16_index


def reference_index(perspective, king_sq, piece_sq, piece_type, piece_colour):
    effective_sq = piece_sq ^ (56 if perspective == 1 else 0)
    effective_colour = piece_colour ^ (1 if perspective == 1 else 0)
    base = ((effective_colour * 6) + piece_type) * 64 + effective_sq

    effective_king_sq = king_sq ^ (56 if perspective == 1 else 0)
    file_ = effective_king_sq & 7
    rank = effective_king_sq >> 3
    file_map = (0, 1, 2, 3, 3, 2, 1, 0)
    bucket = (rank // 2) * 4 + file_map[file_]
    flip = 7 if (king_sq & 7) > 3 else 0
    return bucket * 768 + (base ^ flip)


def test_index_matches_shayveri_runtime_formula_exhaustively():
    for perspective in (0, 1):
        for king_sq in chess.SQUARES:
            for piece_sq in chess.SQUARES:
                for piece_type in range(6):
                    for piece_colour in (0, 1):
                        actual = shayveri_kb16_index(
                            perspective,
                            king_sq,
                            piece_sq,
                            piece_type,
                            piece_colour,
                        )
                        expected = reference_index(
                            perspective,
                            king_sq,
                            piece_sq,
                            piece_type,
                            piece_colour,
                        )
                        assert actual == expected
                        assert 0 <= actual < 16 * 768


def test_cpp_loader_matches_python_indices():
    fen = "r3k2r/pp1n1ppp/2p1bn2/3qp3/3P4/2N1PN2/PPQ2PPP/R3K2R w KQkq - 4 12"
    board = chess.Board(fen)
    batch = stream.get_sparse_batch_from_fens(
        "ShayveriKB16", [fen], [0], [23], [0]
    )
    assert batch

    try:
        (
            _us,
            _them,
            white_indices,
            black_indices,
            _outcome,
            _score,
            _piece_count,
        ) = batch.contents.get_tensors("cpu")
    finally:
        stream.destroy_sparse_batch(batch)

    actual_white = {int(i) for i in white_indices[0] if int(i) >= 0}
    actual_black = {int(i) for i in black_indices[0] if int(i) >= 0}

    expected_white = set()
    expected_black = set()
    white_king = board.king(chess.WHITE)
    black_king = board.king(chess.BLACK)
    assert white_king is not None and black_king is not None

    for square, piece in board.piece_map().items():
        piece_type = piece.piece_type - 1
        piece_colour = 0 if piece.color == chess.WHITE else 1
        expected_white.add(
            shayveri_kb16_index(
                0, white_king, square, piece_type, piece_colour
            )
        )
        expected_black.add(
            shayveri_kb16_index(
                1, black_king, square, piece_type, piece_colour
            )
        )

    assert actual_white == expected_white
    assert actual_black == expected_black


def test_primer_bullet_file_decodes_exactly(tmp_path):
    fen = "4k3/pp3ppp/2n5/3p4/3P4/2N1P3/PP3PPP/4K3 w - - 0 20"
    board = chess.Board(fen)
    occupancy = 0
    piece_codes = []
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        occupancy |= 1 << square
        piece_codes.append(
            piece.piece_type - 1 + (0 if piece.color == chess.WHITE else 8)
        )

    packed_pieces = bytearray(16)
    for index, code in enumerate(piece_codes):
        packed_pieces[index // 2] |= code << (4 * (index % 2))

    record = struct.pack(
        "<Q16shBBB3s",
        occupancy,
        bytes(packed_pieces),
        -137,
        2,
        board.king(chess.WHITE),
        board.king(chess.BLACK),
        b"\x7f\x00\x00",
    )
    assert len(record) == 32
    path = tmp_path / "sample.bullet"
    path.write_bytes(record)

    config = DataloaderSkipConfig(
        filtered=False,
        wld_filtered=False,
        soft_early_fen_skipping=0,
    )
    provider = SparseBatchProvider(
        "ShayveriKB16",
        [str(path)],
        batch_size=1,
        cyclic=False,
        num_workers=1,
        config=config,
    )
    (
        us,
        them,
        white_indices,
        black_indices,
        outcome,
        score,
        piece_count,
    ) = next(provider)

    assert us.item() == 1.0
    assert them.item() == 0.0
    assert outcome.item() == 1.0
    assert score.item() == -137
    assert piece_count.item() == len(piece_codes)

    expected_white = set()
    expected_black = set()
    for square, piece in board.piece_map().items():
        piece_type = piece.piece_type - 1
        piece_colour = 0 if piece.color == chess.WHITE else 1
        expected_white.add(
            shayveri_kb16_index(
                0, board.king(chess.WHITE), square, piece_type, piece_colour
            )
        )
        expected_black.add(
            shayveri_kb16_index(
                1, board.king(chess.BLACK), square, piece_type, piece_colour
            )
        )

    actual_white = {int(i) for i in white_indices[0] if int(i) >= 0}
    actual_black = {int(i) for i in black_indices[0] if int(i) >= 0}
    assert actual_white == expected_white
    assert actual_black == expected_black
