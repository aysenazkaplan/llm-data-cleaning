
import hashlib
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
            retry_failed_rows=True
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
            "version": 3,
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

            self.stats["input_tokens"] += (
                call.get("input_tokens", 0)
            )

            self.stats["output_tokens"] += (
                call.get("output_tokens", 0)
            )

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

    def __init__(self, client, task):
        self.client = client
        self.task = task

    def generate(self, demos):
        contract = {
            "input": {
                col: "raw value"
                for col in self.task.columns
            },
            "output": {
                col: (
                    "float or None"
                    if self.task.numeric
                    else "str or None"
                )
                for col in self.task.columns
            }
        }

        prompt = json.dumps(
            {
                "task": self.task.instruction,
                "demonstrations": demo_payload(demos),
                "function_name": "clean_row",
                "contract": contract
            },
            ensure_ascii=False
        )

        system = """
You are an expert Python data cleaning
operator generator.

Generate a generalizable deterministic
Python data cleaning operator.

STRICT REQUIREMENTS:
1. Return executable Python source code only.
2. Do not include Markdown or explanations.
3. Define exactly one function clean_row(row).
4. Input and output must be dictionaries.
5. Return all required target columns.
6. Handle None, invalid types and empty strings.
7. Use only standard-library re and math.
8. Do not access files or networks.
9. Do not use eval, exec, or dynamic imports.
10. Do not hardcode demonstration examples.
11. Preserve correct data.
12. Avoid unsupported corrections.
13. Use try-except when needed.
14. Return code that runs without editing.
"""

        response = self.client.ask(
            system,
            prompt,
            "synthesis"
        )

        return response.text
