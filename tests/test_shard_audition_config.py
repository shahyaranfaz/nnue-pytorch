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
        epoch_size = int(re.search(r"readonly EPOCH_SIZE=(\d+)", text).group(1))
        max_segments = int(re.search(r"readonly MAX_SEGMENTS=(\d+)", text).group(1))
        lanes = re.findall(r"dh2010pc(?:16|19|22|25)\) readonly LANE=(lr_[0-9]+)", text)

        self.assertEqual(lanes, ["lr_220", "lr_310", "lr_4375", "lr_620"])
        self.assertEqual(epoch_size * max_segments, 799_866_880)
        self.assertEqual((epoch_size // batch_size) * max_segments, full_steps)
        self.assertEqual(full_steps, 48_820)

    def test_lr_is_the_only_experimental_variable(self):
        text = WORKER.read_text(encoding="utf-8")
        values = re.findall(
            r"dh2010pc(?:16|19|22|25)\) readonly LANE=lr_[0-9]+; readonly LR=([0-9.]+)",
            text,
        )
        self.assertEqual(values, ["0.0002200", "0.0003100", "0.0004375", "0.0006200"])
        self.assertRegex(text, r"readonly LAMBDA=0\.74\b")
        self.assertIn("/v2_11/parents/v2_5_factorized.pt", text)
        self.assertNotIn("net1_35B_factorized.pt", text)
        self.assertNotIn("--no-wld-filtered", text)

    def test_worker_can_recover_checkpoint_before_ack_window(self):
        text = WORKER.read_text(encoding="utf-8")
        pending_publish = 'mv -- "$pending_file.partial" "$pending_file"'
        training = 'python -u train.py "${args[@]}"'
        self.assertIn(pending_publish, text)
        self.assertLess(text.rindex(pending_publish), text.index(training))
        self.assertIn('actual_step=$(checkpoint_global_step "$checkpoint")', text)
        self.assertIn('publish_completion "$pending_segment" "$pending_shard" 1', text)
        self.assertIn('rm -f -- "$pending_file"', text)
        checkpoint_publish = 'mv -- "$lane_root/checkpoints/current.ckpt.partial" "$checkpoint"'
        failpoint = '"${V211_FAIL_AFTER_CHECKPOINT:-0}" == 1'
        normal_completion = 'publish_completion "$next_segment" "$name"'
        self.assertLess(text.index(checkpoint_publish), text.index(failpoint))
        self.assertLess(text.index(failpoint), text.index(normal_completion))

    def test_worker_exports_only_midpoint_and_final_nnue_gates(self):
        text = WORKER.read_text(encoding="utf-8")
        self.assertIn("completed_segment != half_segment", text)
        self.assertIn("completed_segment != MAX_SEGMENTS", text)
        self.assertIn('python serialize.py "$checkpoint"', text)
        self.assertIn('sha256sum "$output"', text)

    def test_feeder_cycle_and_quota_cap_match_audition(self):
        text = FEEDER.read_text(encoding="utf-8")
        self.assertRegex(text, r"readonly MAX_IN_FLIGHT=1\b")
        self.assertNotIn('\n        STATE="$state"', text)
        self.assertIn('env STATE="$state"', text)
        self.assertIn('echo ready-meta', text)
        self.assertIn('[[ "$remote_status" == ready-no-meta ]]', text)
        self.assertIn("Remote inventory failed; retrying", text)
        self.assertIn('prepared_file="$STREAM/prepared.env"', text)
        self.assertIn('Publishing prepared successor', text)
        self.assertIn('Preparing local successor', text)
        self.assertLess(text.index('Publishing prepared successor'), text.index('Preparing local successor'))
        schedule = re.search(r"readonly SCHEDULE=\(([^)]+)\)", text).group(1).split()
        self.assertEqual(schedule, ["v210"] * 10)
        self.assertIn("LR-screen feeder complete", text)
        self.assertIn("schedule_index >= ${#SCHEDULE[@]}", text)
        self.assertNotIn("stockfish_base", text)
        self.assertNotIn("t80_base", text)

    def test_lr_screen_uses_fresh_persistent_and_local_namespaces(self):
        worker = WORKER.read_text(encoding="utf-8")
        feeder = FEEDER.read_text(encoding="utf-8")
        self.assertIn("/student/anfazsha/v2_11_lr_screen", worker)
        self.assertIn("/tmp/anfazsha-v211-lr-screen", worker)
        self.assertIn("/mnt/d/nnue/v211_lr_screen_stream", feeder)
        self.assertIn("/student/anfazsha/v2_11_lr_screen", feeder)

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
