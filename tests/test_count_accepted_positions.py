import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

from scripts.shard_stream.count_accepted_positions import count_profile


class FakeConfig:
    def __init__(self):
        self.wld_filtered = True
        self.soft_early_fen_skipping = 20


class CountAcceptedPositionsTest(unittest.TestCase):
    def run_profile(self, profile):
        observed = {}

        def dataset(feature_set, filenames, batch_size, **kwargs):
            observed.update(
                feature_set=feature_set,
                filenames=filenames,
                batch_size=batch_size,
                kwargs=kwargs,
            )
            return iter((object(), object(), object()))

        fake_loader = types.SimpleNamespace(
            DataloaderSkipConfig=FakeConfig,
            SparseBatchDataset=dataset,
        )
        with patch.dict(sys.modules, {"data_loader": fake_loader}):
            result = count_profile([Path("sample.binpack")], profile, 16_384, 2)
        return observed, result

    def test_lane_a_preserves_v210_net1_filters(self):
        observed, result = self.run_profile("lane-a")
        config = observed["kwargs"]["config"]
        self.assertTrue(config.wld_filtered)
        self.assertEqual(config.soft_early_fen_skipping, 20)
        self.assertFalse(observed["kwargs"]["cyclic"])
        self.assertEqual(result["accepted_positions"], 49_152)

    def test_modern_profile_disables_v211_filters(self):
        observed, result = self.run_profile("modern")
        config = observed["kwargs"]["config"]
        self.assertFalse(config.wld_filtered)
        self.assertEqual(config.soft_early_fen_skipping, -1)
        self.assertFalse(observed["kwargs"]["cyclic"])
        self.assertEqual(result["complete_batches"], 3)


if __name__ == "__main__":
    unittest.main()
