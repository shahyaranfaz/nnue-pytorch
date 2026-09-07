import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "shard_stream" / "lab_worker.sh"
FEEDER = ROOT / "scripts" / "shard_stream" / "source_feeder.sh"
SMOKE = ROOT / "scripts" / "shard_stream" / "smoke_lab_worker.sh"


class ShardAuditionConfigTest(unittest.TestCase):
    def test_all_lanes_have_equal_budget_and_scheduler_horizon(self):
        text = WORKER.read_text(encoding="utf-8")
        batch_size = int(re.search(r"readonly BATCH_SIZE=(\d+)", text).group(1))
        full_steps = int(re.search(r"readonly FULL_STEPS=(\d+)", text).group(1))
        lane_values = re.findall(
            r"dh2010pc(?:16|19|22|25)\).*?EPOCH_SIZE=(\d+); readonly MAX_SEGMENTS=(\d+);",
            text,
        )

        self.assertEqual(len(lane_values), 4)
        budgets = {int(epoch_size) * int(segments) for epoch_size, segments in lane_values}
        steps = {
            (int(epoch_size) // batch_size) * int(segments)
            for epoch_size, segments in lane_values
        }
        self.assertEqual(budgets, {799_866_880})
        self.assertEqual(steps, {full_steps})
        self.assertEqual(full_steps, 48_820)

    def test_feeder_cycle_and_quota_cap_match_audition(self):
        text = FEEDER.read_text(encoding="utf-8")
        self.assertRegex(text, r"readonly MAX_IN_FLIGHT=1\b")
        schedule = re.search(r"readonly SCHEDULE=\(([^)]+)\)", text).group(1).split()
        self.assertEqual(len(schedule), 20)
        self.assertEqual(schedule.count("v210"), 10)
        self.assertEqual(schedule.count("stockfish"), 7)
        self.assertEqual(schedule.count("t80"), 3)

    def test_smoke_is_disposable_and_checks_two_phase_resume(self):
        text = SMOKE.read_text(encoding="utf-8")
        self.assertRegex(text, r"readonly EPOCH_SIZE=1048576\b")
        self.assertRegex(text, r"readonly FULL_STEPS=128\b")
        self.assertIn("for target_epoch in 1 2", text)
        self.assertIn('--resume-from-checkpoint="$checkpoint"', text)
        self.assertIn('checkpoint.get("global_step")', text)
        self.assertIn('checkpoint.get("optimizer_states")', text)
        self.assertIn('checkpoint.get("lr_schedulers")', text)
        self.assertIn("cmp -- \"$net\" \"$roundtrip\"", text)
        self.assertNotIn("$ROOT/acks", text)
        self.assertNotIn("$ROOT/lanes", text)


if __name__ == "__main__":
    unittest.main()
