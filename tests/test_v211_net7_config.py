import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "shayveri_v211_net7_train.sh"


class Net7ConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_budget_and_epoch_gates_are_exact(self):
        batch = int(re.search(r"readonly BATCH_SIZE=(\d+)", self.text).group(1))
        epoch = int(re.search(r"readonly EPOCH_SIZE=(\d+)", self.text).group(1))
        epochs = int(re.search(r"readonly EPOCHS=(\d+)", self.text).group(1))
        steps = int(re.search(r"readonly ONE_CYCLE_STEPS=(\d+)", self.text).group(1))
        self.assertEqual(epoch % batch, 0)
        self.assertEqual(epoch // batch, 30_517)
        self.assertEqual(steps, 305_170)
        self.assertEqual(epoch * epochs, steps * batch)
        self.assertEqual(epoch * epochs, 4_999_905_280)

    def test_recipe_is_net6b_scaled_to_five_billion(self):
        self.assertIn("net1_35B_factorized.pt", self.text)
        self.assertIn("readonly MAX_LR=0.000020", self.text)
        self.assertIn("--lambda=0.74 --start-lambda=0.74 --end-lambda=0.74", self.text)
        self.assertIn("--no-wld-filtered --soft-early-fen-skipping=-1", self.text)
        self.assertIn("readonly SEED=42", self.text)
        self.assertIn("${#FILES[@]}\" -eq 5", self.text)
        self.assertNotIn("secondary-datasets", self.text)

    def test_every_epoch_exports_and_run_is_non_overwriting(self):
        self.assertIn("--network-save-period=1 --save-top-k=-1", self.text)
        self.assertIn("refusing to overwrite run", self.text)
        self.assertIn("refusing to overwrite preflight", self.text)
        self.assertIn("recover) [[ $# -eq 2 ]]", self.text)
        self.assertIn("cmp -- \"$net\" \"$roundtrip\"", self.text)
        self.assertIn("go nodes 1000", self.text)


if __name__ == "__main__":
    unittest.main()
