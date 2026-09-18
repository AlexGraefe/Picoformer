import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

from picoformer.evaluate_checkpoints import (
    SUITES,
    _run_and_log,
    discover_checkpoints,
    harness_command,
    requested_suites,
)
from picoformer.plot_evaluations import load_metrics, plot_evaluations


class EvaluateCheckpointsTest(unittest.TestCase):
    def test_discovery_groups_runs_and_converts_zero_based_steps(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for run, steps in (("run_a", (99, 199)), ("run_b", (49,))):
                for step in steps:
                    (root / run / "checkpoints" / f"epoch_0_step_{step}" / "model").mkdir(
                        parents=True
                    )
            checkpoints = discover_checkpoints([root])
            self.assertEqual(
                [(item.run_name, item.training_step) for item in checkpoints],
                [("run_a", 100), ("run_a", 200), ("run_b", 50)],
            )

    def test_command_matches_readme_protocol(self) -> None:
        command = harness_command(
            executable="lm-eval",
            model_path=Path("/model"),
            tokenizer="/tokenizer",
            dtype="bfloat16",
            tasks=SUITES["instruct"],
            suite="instruct",
            output_path=Path("/results"),
            device="cuda:0",
            batch_size="auto",
            phase="smoke",
            smoke_limit=20,
        )
        self.assertEqual(command[:2], ["lm-eval", "run"])
        self.assertIn("--apply_chat_template", command)
        self.assertIn("--log_samples", command)
        self.assertIn("--write_out", command)
        self.assertEqual(command[command.index("--limit") + 1], "20")
        self.assertEqual(requested_suites("instruct"), ("instruct", "instruct-retention"))

    def test_load_and_plot_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            evaluation = root / "run" / "epoch_0_step_99" / "pretrain" / "full"
            evaluation.mkdir(parents=True)
            manifest = {
                "checkpoint": "/run/checkpoints/epoch_0_step_99",
                "phase": "full",
                "run_name": "run",
                "status": "complete",
                "suite": "pretrain",
                "training_step": 100,
            }
            (evaluation / "evaluation.json").write_text(json.dumps(manifest), encoding="utf-8")
            result = {
                "results": {
                    "hellaswag": {
                        "alias": "hellaswag",
                        "acc_norm,none": 0.42,
                        "acc_norm_stderr,none": 0.01,
                    }
                }
            }
            (evaluation / "results_test.json").write_text(json.dumps(result), encoding="utf-8")

            rows = load_metrics(root)
            self.assertEqual(len(rows), 1)
            self.assertEqual((rows[0].step, rows[0].value, rows[0].stderr), (100, 0.42, 0.01))
            outputs = plot_evaluations(root)
            self.assertTrue(all(output.is_file() for output in outputs))
            self.assertEqual(outputs[0].name, "metrics-full.csv")

    def test_run_and_log_streams_carriage_returns_and_heartbeats(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "command.log"
            output = io.StringIO()
            child = (
                "import sys,time; "
                "sys.stdout.write('download 10%' + chr(13)); "
                "sys.stdout.flush(); "
                "time.sleep(0.15); "
                "print('download 100%')"
            )
            with contextlib.redirect_stdout(output):
                return_code = _run_and_log(
                    [sys.executable, "-c", child],
                    log_path,
                    heartbeat_seconds=0.05,
                )

            self.assertEqual(return_code, 0)
            self.assertIn("download 10%\r", output.getvalue())
            self.assertIn("lm-eval is still running", output.getvalue())
            self.assertIn("download 100%", log_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
