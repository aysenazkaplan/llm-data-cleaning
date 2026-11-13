
import hashlib
import ast
import json
import math
import os
import time

from pathlib import Path

from cleaning.llm import RateLimitExhausted


def demo_payload(demos):
    return [
        {
            "dirty": row.dirty,
            "clean": row.clean
        }
        for row in demos
    ]


def task_context(task, demos, schema=None):

    characteristics = {}
    for column in task.columns:
        values = [row.dirty.get(column) for row in demos]
        present = [value for value in values if value is not None]
        characteristics[column] = {
            "observed_types": sorted({type(value).__name__ for value in present}),
            "missing_count": len(values) - len(present),
            "sample_values": list(dict.fromkeys(present))[:10],
        }
    return {
        "schema": schema or {col: "raw value" for col in task.columns},
        "input_columns": list(task.columns),
        "target_columns": list(task.columns),
        "output_types": {col: "number or null" if task.numeric else "string or null"
                         for col in task.columns},
        "data_characteristics": characteristics,
        "user_requirements": task.instruction,
    }


def normalize_output(obj, task):

    if not isinstance(obj, dict):
        raise ValueError(
            "Output must be a JSON object"
        )

    result = {}

    for column in task.columns:
        if column not in obj:
            raise ValueError(
                f"Missing column: {column}"
            )

        value = obj[column]

        if value is None:
            result[column] = None

        elif task.numeric:
            if isinstance(value, bool):
                raise ValueError(
                    "Boolean is not a valid number"
                )

            try:
                number = float(value)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid numeric value for "
                    f"{column}: {value!r}"
                ) from exc

            if not math.isfinite(number):
                raise ValueError(
                    "Non-finite number"
                )

            result[column] = number

        else:
            if not isinstance(value, str):
                raise ValueError(
                    f"{column} must be a string or null"
                )

            result[column] = value

    return result


def extract_json_objects(raw):

    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(
            "Empty LLM response"
        )

    decoder = json.JSONDecoder()

    objects = []
    position = 0

    while position < len(raw):
        start = raw.find("{", position)

        if start == -1:
            break

        try:
            obj, consumed = decoder.raw_decode(
                raw[start:]
            )

            objects.append(obj)
            position = start + consumed

        except json.JSONDecodeError:
            position = start + 1

    if not objects:
        raise ValueError(
            "No valid JSON object found"
        )

    return objects


def parse_direct_results(raw, task):

    items = []

    for obj in extract_json_objects(raw):
        if not isinstance(obj, dict):
            continue

        if isinstance(obj.get("results"), list):
            items.extend(obj["results"])

        elif (
                "index" in obj
                and all(
            col in obj
            for col in task.columns
        )
        ):
            items.append(obj)

    if not items:
        raise ValueError(
            "No usable cleaning results found"
        )

    return items


class DirectLLMCleaner:
    def __init__(
            self,
            client,
            task,
            batch_size=10,
            checkpoint_path=None,
            reset_checkpoint=False,
            retry_failed_rows=True,
            schema=None
    ):
        if batch_size < 1:
            raise ValueError(
                "batch_size must be positive"
            )

        self.client = client
        self.task = task
        self.batch_size = batch_size

        self.checkpoint_path = (
            Path(checkpoint_path)
            if checkpoint_path is not None
            else None
        )

        self.reset_checkpoint = reset_checkpoint
        self.retry_failed_rows = retry_failed_rows
        self.schema = schema

        self.stats = {
            "api_calls": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "elapsed_seconds": 0.0
        }

        log_path = os.getenv(
            "LLM_RAW_LOG_PATH", ""
        ).strip()

        self.raw_log_path = (
            Path(log_path)
            if log_path
            else None
        )

    def _fingerprint(self, demos, evaluation):

        payload = {
            "version": 4,
            "context": task_context(self.task, demos, self.schema),
            "model": self.client.model,
            "base_url": str(self.client.base_url),
            "max_tokens": self.client.max_tokens,
            "task": {
                "name": self.task.name,
                "dataset": self.task.dataset,
                "columns": list(self.task.columns),
                "numeric": self.task.numeric,
                "instruction": self.task.instruction
            },
            "batch_size": self.batch_size,
            "retry_failed_rows": self.retry_failed_rows,
            "demonstrations": demo_payload(demos),
            "evaluation": [
                {
                    "index": row.row_id,
                    "dirty": row.dirty
                }
                for row in evaluation
            ]
        }

        serialized = json.dumps(
            payload,
            sort_keys=True,
            ensure_ascii=False
        )

        return hashlib.sha256(
            serialized.encode("utf-8")
        ).hexdigest()

    def _load_checkpoint(self, fingerprint):
        if (
                self.checkpoint_path is None
                or self.reset_checkpoint
                or not self.checkpoint_path.exists()
        ):
            return {}, {}, set()

        state = json.loads(
            self.checkpoint_path.read_text(
                encoding="utf-8"
            )
        )

        if state.get("fingerprint") != fingerprint:
            raise ValueError(
                "Checkpoint settings do not match "
                "the current experiment.\n"
                f"Checkpoint: {self.checkpoint_path}\n"
                "Use the original experiment settings "
                "or --reset-checkpoint to restart."
            )

        predictions = {
            int(key): value
            for key, value in state.get(
                "predictions", {}
            ).items()
        }

        failures = {
            int(key): value
            for key, value in state.get(
                "failures", {}
            ).items()
        }

        completed = set(
            state.get("completed_batches", [])
        )

        self.stats.update(
            state.get("stats", {})
        )

        return predictions, failures, completed

    def _save_checkpoint(
            self,
            fingerprint,
            predictions,
            failures,
            completed
    ):
        if self.checkpoint_path is None:
            return

        self.checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        state = {
            "fingerprint": fingerprint,
            "completed_batches": sorted(completed),
            "predictions": predictions,
            "failures": failures,
            "stats": self.stats
        }

        temp_path = self.checkpoint_path.with_name(
            self.checkpoint_path.name + ".tmp"
        )

        try:
            with temp_path.open(
                    "w",
                    encoding="utf-8"
            ) as file:
                json.dump(
                    state,
                    file,
                    ensure_ascii=False
                )
                file.flush()
                os.fsync(file.fileno())

            os.replace(
                temp_path,
                self.checkpoint_path
            )

        finally:
            if temp_path.exists():
                temp_path.unlink()

    def _log_response(
            self,
            indices,
            phase,
            raw=None,
            error=None
    ):


        if self.raw_log_path is None:
            return

        self.raw_log_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        entry = {
            "task": self.task.name,
            "dataset": self.task.dataset,
            "phase": phase,
            "indices": indices,
            "raw_response": raw,
            "error": error
        }

        with self.raw_log_path.open(
                "a",
                encoding="utf-8"
        ) as file:
            file.write(
                json.dumps(
                    entry,
                    ensure_ascii=False
                ) + "\n"
            )

    def _prompts(self, batch, demos):
        system = """
You are a tabular data cleaning assistant.

Clean the input records using the task
instruction and dirty-clean demonstrations.

STRICT OUTPUT FORMAT:
1. Return only one valid JSON object.
2. The object must contain a "results" array.
3. Return one result per input record.
4. Preserve the original "index".
5. Return all required target columns.
6. Use JSON null only for genuinely missing
   or unknown values.
7. Do not omit or add records.
8. Do not include Markdown or explanations.
9. Preserve already-correct values.
10. Correct errors supported by the examples.
11. Avoid inventing missing information.
12. Preserve the intended meaning and units.
"""

        example_result = {
            "index": 123,
            **{
                col: (
                    12.0
                    if self.task.numeric
                    else "example value"
                )
                for col in self.task.columns
            }
        }

        user = json.dumps(
            {
                "cleaning_task": self.task.instruction,
                "context": task_context(self.task, demos, self.schema),
                "demonstrations": demo_payload(demos),
                "input_records": [
                    {
                        "index": row.row_id,
                        **row.dirty
                    }
                    for row in batch
                ],
                "required_output_format": {
                    "results": [example_result]
                }
            },
            ensure_ascii=False
        )

        return system, user

    def _request(self, batch, demos, phase):
        system, user = self._prompts(
            batch, demos
        )

        indices = [
            row.row_id for row in batch
        ]

        try:
            response = self.client.ask(
                system,
                user,
                "direct"
            )

            self._log_response(
                indices,
                phase,
                raw=response.text
            )

            return response.text

        except Exception as exc:
            self._log_response(
                indices,
                phase,
                error=(
                    f"{type(exc).__name__}: {exc}"
                )
            )
            raise

    def _parse(self, raw, batch):


        expected = {
            row.row_id for row in batch
        }

        try:
            items = parse_direct_results(
                raw, self.task
            )

        except Exception as exc:
            return {}, {
                idx: (
                    f"{type(exc).__name__}: {exc}"
                )
                for idx in expected
            }

        candidates = {}

        for item in items:
            if not isinstance(item, dict):
                continue

            try:
                idx = int(item["index"])
            except (
                    KeyError,
                    ValueError,
                    TypeError
            ):
                continue

            if idx in expected:
                candidates.setdefault(
                    idx, []
                ).append(item)

        valid = {}
        errors = {}

        for idx in expected:
            options = candidates.get(idx, [])

            if not options:
                errors[idx] = (
                    "Missing result for index"
                )
                continue

            if len(options) > 1:
                errors[idx] = (
                    "Duplicate result for index"
                )
                continue

            try:
                valid[idx] = normalize_output(
                    options[0],
                    self.task
                )

            except (
                    ValueError,
                    TypeError,
                    OverflowError
            ) as exc:
                errors[idx] = (
                    f"{type(exc).__name__}: {exc}"
                )

        return valid, errors

    def _collect_usage(self, start_index):
        new_calls = self.client.calls[
            start_index:
        ]

        for call in new_calls:
            self.stats["api_calls"] += 1
            for key in ("input_tokens", "output_tokens"):
                value = call.get(key)
                if (not call.get("usage_available", True) or value is None
                        or self.stats[key] is None):
                    self.stats[key] = None
                else:
                    self.stats[key] += value

    def run(self, demos, evaluation):
        fingerprint = self._fingerprint(
            demos,
            evaluation
        )

        predictions, failures, completed = (
            self._load_checkpoint(fingerprint)
        )

        total_batches = math.ceil(
            len(evaluation) / self.batch_size
        )

        print(
            f"[Checkpoint] "
            f"{len(completed)}/{total_batches} "
            "batches completed.",
            flush=True
        )

        for batch_number, start in enumerate(
                range(
                    0,
                    len(evaluation),
                    self.batch_size
                )
        ):
            if batch_number in completed:
                continue

            batch = evaluation[
                start:start + self.batch_size
            ]

            print(
                f"[Direct LLM] "
                f"Batch {batch_number + 1}/"
                f"{total_batches}",
                flush=True
            )

            calls_before = len(self.client.calls)
            batch_started = time.perf_counter()

            try:
                raw = self._request(
                    batch,
                    demos,
                    phase="batch"
                )

                valid, errors = self._parse(
                    raw,
                    batch
                )

                if (
                        self.retry_failed_rows
                        and errors
                ):
                    for row in batch:
                        idx = row.row_id

                        if idx not in errors:
                            continue

                        retry_raw = self._request(
                            [row],
                            demos,
                            phase="single_retry"
                        )

                        recovered, problems = self._parse(
                            retry_raw,
                            [row]
                        )

                        if idx in recovered:
                            valid[idx] = recovered[idx]
                            errors.pop(idx, None)

                        else:
                            errors[idx] = problems.get(
                                idx,
                                "Invalid single-record response"
                            )

            except (
                    KeyboardInterrupt,
                    RateLimitExhausted
            ):
                self._collect_usage(calls_before)

                self.stats["elapsed_seconds"] += (
                        time.perf_counter()
                        - batch_started
                )

                self._save_checkpoint(
                    fingerprint,
                    predictions,
                    failures,
                    completed
                )

                print(
                    "[Checkpoint] Experiment paused. "
                    "Completed batches are saved.",
                    flush=True
                )

                raise

            except Exception as exc:
                self._collect_usage(calls_before)

                self.stats["elapsed_seconds"] += (
                        time.perf_counter()
                        - batch_started
                )

                self._save_checkpoint(
                    fingerprint,
                    predictions,
                    failures,
                    completed
                )

                raise RuntimeError(
                    f"Batch {batch_number + 1} "
                    "was interrupted. Checkpoint "
                    "preserved; rerun to resume."
                ) from exc

            self._collect_usage(calls_before)

            self.stats["elapsed_seconds"] += (
                    time.perf_counter()
                    - batch_started
            )

            predictions.update(valid)

            for idx in valid:
                failures.pop(idx, None)

            failures.update(errors)

            for row in batch:
                idx = row.row_id

                if idx not in valid:
                    predictions[idx] = None

                    failures.setdefault(
                        idx,
                        "No valid prediction"
                    )

            completed.add(batch_number)

            self._save_checkpoint(
                fingerprint,
                predictions,
                failures,
                completed
            )

        if len(completed) != total_batches:
            raise RuntimeError(
                "Experiment is incomplete."
            )

        return predictions, failures


class OperatorSynthesizer:


    def __init__(self, client, task, schema=None):
        self.client = client
        self.task = task
        self.schema = schema
        self.last_source = None

    def _payload(self, demos, previous_source=None, feedback=None):
        payload = {
            "context": task_context(self.task, demos, self.schema),
            "demonstrations": demo_payload(demos),
        }
        if feedback is not None:
            payload.update(
                previous_code=previous_source,
                validation_feedback=feedback,
                feedback_source="held-out validation only; no final evaluation labels",
                repair_goal="Correct demonstrated failures without changing valid outputs or interface"
            )
        return payload

    @staticmethod
    def _extract_code(text):
        source = text.strip()
        if source.startswith("```") and source.endswith("```"):
            source = "\n".join(source.splitlines()[1:-1])
        return source

    def generate(self, demos, previous_source=None, feedback=None):
        payload = self._payload(demos, previous_source, feedback)
        payload["function_name"] = "clean_row(row)"
        response = self.client.ask(
            "Generate a generalizable deterministic Python cleaning operator. "
            "Return executable Python code only. Define clean_row(row), taking a dictionary "
            "and returning exactly the target columns in the requested types. You own the "
            "entire implementation. Handle None, empty strings and invalid values. "
            "Use field relationships when relevant. Preserve correct values and avoid unsupported "
            "corrections. Infer rules, not a lookup table of demonstration records. "
            "Use only standard-library re/math imports. No files, networks, tools, eval, exec "
            "or dynamic imports. If validation feedback is provided, fix the previous code.",
            json.dumps(payload, ensure_ascii=False),
            "synthesis_repair" if feedback is not None else "synthesis",
        )
        self.last_source = self._extract_code(response.text)
        return self.last_source


class GuidedOperatorSynthesizer(OperatorSynthesizer):


    def template(self):
        return '''import math

# LLM_LOGIC

def clean_row(row):
    if not isinstance(row, dict):
        raise ValueError("Input must be a dictionary")
    columns = TARGET_COLUMNS
    values = {column: row.get(column) for column in columns}
    # Pass a copy: abstaining must preserve the original input even after mutation.
    candidate = clean_fields(dict(values))
    if candidate is None:
        candidate = values
    if not isinstance(candidate, dict) or set(candidate) != set(columns):
        raise ValueError("Return exactly the target columns, or None to abstain")
    output = {}
    for column in columns:
        value = candidate[column]
        if value is None:
            output[column] = None
        elif NUMERIC_OUTPUT:
            if isinstance(value, bool):
                raise ValueError("Boolean is not a number")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("Non-finite number")
            output[column] = number
        elif isinstance(value, str):
            output[column] = value
        else:
            raise ValueError("Expected string or None")
    return output
'''.replace("TARGET_COLUMNS", repr(tuple(self.task.columns))).replace(
            "NUMERIC_OUTPUT", repr(self.task.numeric))

    def assemble(self, text):
        from cleaning.operator import OperatorValidator

        source = self._extract_code(text)
        self.last_source = source
        tree = ast.parse(source)
        functions = []
        for node in tree.body:
            if isinstance(node, ast.Import):
                if any(a.name not in {"re", "math"} or a.asname for a in node.names):
                    raise ValueError("Only unaliased re/math imports allowed")
            elif isinstance(node, ast.FunctionDef) and node.name == "clean_fields":
                functions.append(node)
            else:
                raise ValueError("Generate only clean_fields(values)")
        if len(functions) != 1:
            raise ValueError("Expected one clean_fields function")
        fn = functions[0]
        if ([a.arg for a in fn.args.args] != ["values"] or fn.args.posonlyargs
                or fn.args.kwonlyargs or fn.args.defaults or fn.args.vararg or fn.args.kwarg
                or fn.decorator_list or fn.returns or fn.args.args[0].annotation):
            raise ValueError("Expected clean_fields(values) without extra arguments")
        for node in ast.walk(fn):
            if node is not fn and isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                raise ValueError("Nested declarations are not allowed")

        result = self.template().replace("# LLM_LOGIC", source, 1)
        OperatorValidator.validate(result)
        return result

    def generate(self, demos, previous_source=None, feedback=None):
        payload = self._payload(demos, previous_source, feedback)
        payload["fixed_template"] = self.template()
        payload["logic_contract"] = (
            "clean_fields(values): return a dictionary with every target column, "
            "or None to abstain and preserve the input. Do not modify the fixed wrapper.")
        response = self.client.ask(
            "Complete only the task-specific logic in the fixed template. "
            "Return exactly one function clean_fields(values), optionally importing re or math. "
            "Input contains all target fields; use relationships between fields when relevant. "
            "Infer general rules from schema, characteristics, requirements and demonstrations. "
            "Preserve correct values; abstain when evidence is insufficient. Do not hardcode "
            "example records. Do not generate clean_row, helpers, decorators, annotations, "
            "default arguments or top-level statements. No files, networks, tools, eval, exec "
            "or dynamic imports. If validation feedback is provided, repair only clean_fields. "
            "Return executable Python source only.",
            json.dumps(payload, ensure_ascii=False),
            "guided_repair" if feedback is not None else "guided_synthesis",
        )
        return self.assemble(response.text)

    def validate_operator(self, source):

        tree = ast.parse(source)
        logic = [node for node in tree.body[1:]
                 if isinstance(node, ast.Import)
                 or isinstance(node, ast.FunctionDef) and node.name == "clean_fields"]
        assembled = self.assemble(ast.unparse(ast.Module(body=logic, type_ignores=[])))
        if ast.dump(ast.parse(assembled)) != ast.dump(tree):
            raise ValueError("Cached guided operator does not match the fixed template; regenerate it")
