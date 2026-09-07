import hashlib
import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "shayveri_v211_net4_train.sh"


class V211Net4ConfigTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = SCRIPT.read_text(encoding="utf-8")

    def value(self, name):
        return re.search(rf"readonly {name}=([^\n]+)", self.text).group(1)

    def test_frozen_budget_and_schedule(self):
        epoch_size = int(self.value("EPOCH_SIZE"))
        epochs = int(self.value("EPOCHS"))
        cycle = int(self.value("ONE_CYCLE_STEPS"))
        refinement = int(self.value("REFINEMENT_STEPS"))
        self.assertEqual(epoch_size, 1_000_013_824)
        self.assertEqual(epochs, 80)
        self.assertEqual(epoch_size // 16_384, 61_036)
        self.assertEqual(cycle, 2_441_440)
        self.assertEqual(refinement, 2_441_440)
        self.assertEqual((epoch_size // 16_384) * epochs, cycle + refinement)

    def test_recipe_is_explicit(self):
        self.assertIn("--lambda=0.74 --start-lambda=0.74 --end-lambda=0.74", self.text)
        self.assertIn("--lr=\"$MAX_LR\"", self.text)
        self.assertIn("--one-cycle-warmup-pct=0.2", self.text)
        self.assertIn("--one-cycle-start-div=25", self.text)
        self.assertIn("--one-cycle-final-div=50", self.text)
        self.assertIn("--one-cycle-refinement-lr=\"$REFINEMENT_LR\"", self.text)
        self.assertIn("--network-save-period=5 --save-top-k=-1", self.text)
        self.assertNotIn("--no-wld-filtered", self.text)
        self.assertNotIn("--soft-early-fen-skipping", self.text)

    def test_manifest_and_parent_are_pinned(self):
        self.assertIn("c9b37e262cb917650b445e54d3bb6153d4698b84dcb094c657d75145cd242a79", self.text)
        self.assertIn("bb3ef74734d8a7e41542e6715e44b718d775608c32ce61308f7cc58a12322c56", self.text)
        self.assertEqual(self.text.count("@DATA@/"), 20)
        block = re.search(r"<<'FILES'\n(.*?)\nFILES", self.text, re.DOTALL).group(1)
        manifest = block.replace("@DATA@", "/mnt/d/nnue/robotmoon") + "\n"
        self.assertEqual(
            hashlib.sha256(manifest.encode()).hexdigest(),
            "bb3ef74734d8a7e41542e6715e44b718d775608c32ce61308f7cc58a12322c56",
        )

    def test_preflight_crosses_boundary_and_loads_engine(self):
        self.assertIn("--one-cycle-steps=64 --one-cycle-refinement-steps=64", self.text)
        self.assertIn("for target_epoch in 1 2", self.text)
        self.assertIn("--resume-from-checkpoint=\"$checkpoint\"", self.text)
        self.assertIn("cmp -- \"$net\" \"$roundtrip\"", self.text)
        self.assertIn("go nodes 1000", self.text)

    def test_start_refuses_overwrite(self):
        self.assertIn('[[ ! -e "$RUN_ROOT" ]]', self.text)


if __name__ == "__main__":
    unittest.main()
