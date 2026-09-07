import json
import shutil
import subprocess
import sys
import tempfile
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
        self.assertEqual(observed["feature_set"], "ShayveriKB16")
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

    def test_file_entrypoint_imports_from_repository_root(self):
        source = Path(__file__).resolve().parents[1] / "scripts" / "shard_stream" / "count_accepted_positions.py"
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            script = root / "scripts" / "shard_stream" / source.name
            script.parent.mkdir(parents=True)
            shutil.copyfile(source, script)
            (root / "data_loader.py").write_text(
                "class DataloaderSkipConfig:\n"
                "    wld_filtered = True\n"
                "    soft_early_fen_skipping = 20\n"
                "def SparseBatchDataset(*args, **kwargs):\n"
                "    return iter((1, 2))\n",
                encoding="utf-8",
            )
            sample = root / "sample.binpack"
            sample.write_bytes(b"test")

            completed = subprocess.run(
                [sys.executable, str(script), "--profile=lane-a", str(sample)],
                cwd=root,
                capture_output=True,
                text=True,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            result = json.loads(completed.stdout)
            self.assertEqual(result["counts"][0]["accepted_positions"], 32_768)


if __name__ == "__main__":
    unittest.main()
