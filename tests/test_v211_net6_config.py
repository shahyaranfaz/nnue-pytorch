import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "shayveri_v211_net6_screen.sh"


class Net6ConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def test_budget_is_exactly_799866880_presentations(self):
        batch = int(re.search(r"readonly BATCH_SIZE=(\d+)", self.text).group(1))
        epoch = int(re.search(r"readonly EPOCH_SIZE=(\d+)", self.text).group(1))
        epochs = int(re.search(r"readonly EPOCHS=(\d+)", self.text).group(1))
        steps = int(re.search(r"readonly ONE_CYCLE_STEPS=(\d+)", self.text).group(1))
        self.assertEqual(epoch * epochs, 799_866_880)
        self.assertEqual(steps * batch, 799_866_880)
        self.assertEqual(epoch // batch, 24_410)

    def test_shared_recipe_is_frozen(self):
        self.assertIn("readonly MAX_LR=0.000020", self.text)
        self.assertIn("--lambda=0.74 --start-lambda=0.74 --end-lambda=0.74", self.text)
        self.assertIn("--no-wld-filtered --soft-early-fen-skipping=-1", self.text)
        self.assertIn("readonly SEED=42", self.text)
        self.assertIn("net1_35B_factorized.pt", self.text)
        self.assertIn("abe2ce14392ea07d0f2eb3279468299fc546bbd1a14f5367877d1c1adfe684e1", self.text)

    def test_four_lanes_vary_only_corpus(self):
        self.assertIn("A) echo t80_2023", self.text)
        self.assertIn("B) echo stockfish_new", self.text)
        self.assertIn("C) echo t80_2024", self.text)
        self.assertIn("D) echo mix_9_7_4", self.text)
        self.assertIn("${#T80_2023[@]}\" -eq 12", self.text)
        self.assertIn("${#STOCKFISH_NEW[@]}\" -eq 5", self.text)
        self.assertIn("${#T80_2024[@]}\" -eq 6", self.text)

    def test_control_is_exact_9_7_4(self):
        self.assertIn("--secondary-batches-per-cycle=7", self.text)
        self.assertIn("--tertiary-batches-per-cycle=4", self.text)
        self.assertIn("--mix-cycle-batches=20", self.text)
        self.assertIn("accepted_batch_mix=9/7/4", self.text)

    def test_launcher_is_local_sequential_and_non_overwriting(self):
        self.assertIn("/mnt/d/nnue/runs/v211_net6", self.text)
        self.assertNotIn("ssh ", self.text)
        self.assertNotIn("scp ", self.text)
        self.assertIn('for lane in A B C D; do run_lane "$lane" start; done', self.text)
        self.assertIn("refusing to overwrite incomplete lane", self.text)

    def test_exports_both_gates_and_supports_preflight_recovery(self):
        self.assertIn("for epoch in 0 1", self.text)
        self.assertIn("--network-save-period=1 --save-top-k=-1", self.text)
        self.assertIn("preflight) [[ $# -eq 1 ]]", self.text)
        self.assertIn("recover) [[ $# -eq 3 ]]", self.text)
        self.assertIn("cmp -- \"$net\" \"$roundtrip\"", self.text)
        self.assertIn("go nodes 1000", self.text)


if __name__ == "__main__":
    unittest.main()
