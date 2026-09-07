import ast
import unittest
from pathlib import Path


MODULE = Path(__file__).resolve().parents[1] / "model" / "lightning_module.py"


class CompileBoundaryTest(unittest.TestCase):
    def test_lightning_logging_stays_outside_compiled_step(self):
        tree = ast.parse(MODULE.read_text(encoding="utf-8"))
        nnue = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "NNUE"
        )
        methods = {
            node.name: node
            for node in nnue.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        recorder = methods["_record_step_loss"]
        step = methods["step_"]

        decorators = {ast.unparse(value) for value in recorder.decorator_list}
        self.assertIn("torch.compiler.disable", decorators)
        recorder_calls = {
            ast.unparse(node.func)
            for node in ast.walk(recorder)
            if isinstance(node, ast.Call)
        }
        self.assertIn("self.log", recorder_calls)
        self.assertTrue(any("loss_metrics" in call for call in recorder_calls))

        step_calls = {
            ast.unparse(node.func)
            for node in ast.walk(step)
            if isinstance(node, ast.Call)
        }
        self.assertIn("self._record_step_loss", step_calls)
        self.assertNotIn("self.log", step_calls)


if __name__ == "__main__":
    unittest.main()
