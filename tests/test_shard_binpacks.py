import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "shard_stream" / "shard_binpacks.py"


def chunk(payload: bytes) -> bytes:
    return b"BINP" + struct.pack("<I", len(payload)) + payload


class ShardBinpacksTest(unittest.TestCase):
    def test_splits_only_between_chunks_and_resumes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "a.binpack"
            second = root / "b.binpack"
            first.write_bytes(chunk(b"a" * 5) + chunk(b"b" * 7))
            second.write_bytes(chunk(b"c" * 3))
            output = root / "out"
            state = root / "state.json"
            command = [sys.executable, str(SCRIPT), "next", str(first), str(second),
                       "--output-dir", str(output), "--state", str(state),
                       "--target-size", "20"]

            self.assertEqual(subprocess.run(command).returncode, 0)
            shard0 = output / "v210_00000.binpack"
            self.assertEqual(shard0.read_bytes(), chunk(b"a" * 5))
            self.assertEqual(subprocess.run(command).returncode, 2)
            self.assertEqual(subprocess.run([sys.executable, str(SCRIPT), "retry",
                                             "--state", str(state)]).returncode, 0)
            self.assertEqual(subprocess.run(command).returncode, 0)
            self.assertEqual(shard0.read_bytes(), chunk(b"a" * 5))
            self.assertEqual(subprocess.run([sys.executable, str(SCRIPT), "ack",
                                             "--state", str(state)]).returncode, 0)
            self.assertFalse(shard0.exists())

            self.assertEqual(subprocess.run(command).returncode, 0)
            self.assertEqual((output / "v210_00001.binpack").read_bytes(), chunk(b"b" * 7))
            self.assertEqual(subprocess.run([sys.executable, str(SCRIPT), "ack",
                                             "--state", str(state)]).returncode, 0)
            self.assertEqual(subprocess.run(command).returncode, 0)
            self.assertEqual((output / "v210_00002.binpack").read_bytes(), chunk(b"c" * 3))
            document = json.loads(state.read_text())
            self.assertEqual(document["input_index"], 2)

    def test_allows_two_pending_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "a.binpack"
            source.write_bytes(chunk(b"a" * 5) + chunk(b"b" * 5))
            command = [sys.executable, str(SCRIPT), "next", str(source),
                       "--output-dir", str(root / "out"),
                       "--state", str(root / "state.json"),
                       "--target-size", "13", "--max-pending", "2"]
            self.assertEqual(subprocess.run(command).returncode, 0)
            self.assertEqual(subprocess.run(command).returncode, 0)
            self.assertEqual(subprocess.run(command).returncode, 2)
            state = json.loads((root / "state.json").read_text())
            self.assertEqual(len(state["pending"]), 2)

    def test_rejects_invalid_header(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "bad.binpack"
            source.write_bytes(b"not-binpack")
            result = subprocess.run([sys.executable, str(SCRIPT), "next", str(source),
                                     "--output-dir", str(root / "out"),
                                     "--state", str(root / "state.json")])
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
