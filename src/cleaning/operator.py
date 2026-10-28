

import ast
import json
import subprocess
import sys

from pathlib import Path

from cleaning.cleaners import normalize_output
from cleaning.evaluation import matching


class OperatorValidator:

    @staticmethod
    def validate(source):

        if not source.strip():
            raise ValueError("Empty source")

        try:
            tree = ast.parse(source)

        except SyntaxError as exc:
            raise ValueError(
                f"Invalid Python syntax at line "
                f"{exc.lineno}: {exc.msg}"
            ) from exc

        functions = [
            node
            for node in tree.body
            if (
                    isinstance(node, ast.FunctionDef)
                    and node.name == "clean_row"
            )
        ]

        if len(functions) != 1:
            raise ValueError(
                "Expected one clean_row function"
            )

        fn = functions[0]

        if (
                len(fn.args.args) != 1
                or fn.args.args[0].arg != "row"
                or fn.args.defaults
                or fn.args.vararg
                or fn.args.kwarg
        ):
            raise ValueError(
                "Expected clean_row(row)"
            )

        denied = (
            ast.ImportFrom,
            ast.Global,
            ast.Nonlocal,
            ast.ClassDef,
            ast.AsyncFunctionDef,
            ast.With,
            ast.AsyncWith,
            ast.Lambda
        )

        denied_names = {
            "eval",
            "exec",
            "open",
            "__import__",
            "compile",
            "input",
            "globals",
            "locals",
            "getattr",
            "setattr"
        }

        for node in ast.walk(tree):

            if isinstance(node, denied):
                raise ValueError(
                    f"Forbidden syntax: "
                    f"{type(node).__name__}"
                )

            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name not in {"re", "math"}:
                        raise ValueError(
                            "Only re and math imports allowed"
                        )

            if isinstance(node, ast.Name):
                if node.id in denied_names:
                    raise ValueError(
                        f"Forbidden name: {node.id}"
                    )

            if isinstance(node, ast.Attribute):
                if node.attr.startswith("__"):
                    raise ValueError(
                        "Dunder attributes forbidden"
                    )

        return True

    @staticmethod
    def validate_examples(
            path,
            demos,
            task,
            *,
            allowed=False
    ):
        predictions, failures = (
            OperatorExecutor(path).run(
                demos,
                task,
                allowed=allowed
            )
        )

        passed = sum(
            matching(
                predictions.get(row.row_id),
                row.clean,
                task
            )
            for row in demos
        )

        return {
            "accuracy": passed / len(demos),
            "passed": (
                    passed == len(demos)
                    and not failures
            ),
            "correct": passed,
            "total": len(demos),
            "failures": failures
        }



class OperatorExecutor:

    def __init__(
            self,
            source_path: Path,
            timeout=45
    ):
        self.source_path = Path(source_path)
        self.timeout = timeout

    def run(
            self,
            rows,
            task,
            *,
            allowed=False
    ):
        if not allowed:
            raise ValueError(
                "Review code before --run-generated"
            )

        source = self.source_path.read_text(
            encoding="utf-8"
        )

        OperatorValidator.validate(source)

        payload = json.dumps([
            {
                "index": row.row_id,
                "input": row.dirty
            }
            for row in rows
        ])

        try:
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "cleaning.worker",
                    str(self.source_path.resolve())
                ],
                input=payload,
                text=True,
                capture_output=True,
                timeout=self.timeout
            )

            if proc.returncode:
                raise RuntimeError(
                    proc.stderr.strip()[-1200:]
                    or f"worker exit {proc.returncode}"
                )

            results = json.loads(proc.stdout)

            if not isinstance(results, list):
                raise ValueError(
                    "Worker output must be a list"
                )

            expected = {
                row.row_id for row in rows
            }

            predictions = {}
            failures = {}

            for item in results:
                if (
                        not isinstance(item, dict)
                        or set(item) != {
                    "index", "output", "error"
                }
                ):
                    raise ValueError(
                        "Invalid worker record"
                    )

                idx = int(item["index"])

                if idx not in expected or idx in predictions:
                    raise ValueError(
                        "Duplicate/unknown row ID"
                    )

                if item["error"] is not None:
                    predictions[idx] = None
                    failures[idx] = item["error"]
                    continue

                try:
                    output = item["output"]

                    if (
                            not isinstance(output, dict)
                            or set(output) != set(task.columns)
                    ):
                        raise ValueError(
                            "Wrong output schema"
                        )

                    predictions[idx] = normalize_output(
                        output,
                        task
                    )

                except Exception as exc:
                    predictions[idx] = None
                    failures[idx] = (
                        f"{type(exc).__name__}: {exc}"
                    )

            missing = expected - set(predictions)

            for idx in missing:
                predictions[idx] = None
                failures[idx] = "Missing operator output"

            return predictions, failures

        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"

            return (
                {row.row_id: None for row in rows},
                {row.row_id: reason for row in rows}
            )
