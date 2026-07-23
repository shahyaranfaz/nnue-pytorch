import chess

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
