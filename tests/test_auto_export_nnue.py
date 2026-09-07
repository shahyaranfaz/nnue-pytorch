import ast
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class AutoExportNNUEConfigTest(unittest.TestCase):
    def test_checkpoint_callback_exports_atomically_after_save(self):
        tree = ast.parse((ROOT / "train.py").read_text(encoding="utf-8"))
        callback = next(
            node for node in tree.body
            if isinstance(node, ast.ClassDef) and node.name == "ConsolidatedCheckpoint"
        )
        methods = {
            node.name: node
            for node in callback.body
            if isinstance(node, ast.FunctionDef)
        }
        save = methods["_save_checkpoint"]
        calls = [
            ast.unparse(node.func)
            for node in ast.walk(save)
            if isinstance(node, ast.Call)
        ]
        self.assertIn("super()._save_checkpoint", calls)
        self.assertIn("self._export_nnue", calls)
        self.assertLess(
            calls.index("super()._save_checkpoint"),
            calls.index("self._export_nnue"),
        )

        export_source = ast.unparse(methods["_export_nnue"])
        self.assertIn("M.ShayveriNNUEWriter", export_source)
        self.assertNotIn("M.NNUEWriter", export_source)
        self.assertIn("os.fsync", export_source)
        self.assertIn("os.replace", export_source)

    def test_export_has_no_disable_flag(self):
        config_source = (ROOT / "config.py").read_text(encoding="utf-8")
        train_source = (ROOT / "train.py").read_text(encoding="utf-8")
        self.assertNotIn("auto_export_nnue", config_source)
        self.assertNotIn("auto_export_nnue", train_source)


if __name__ == "__main__":
    unittest.main()
