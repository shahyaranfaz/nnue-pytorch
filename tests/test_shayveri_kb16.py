import chess

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
