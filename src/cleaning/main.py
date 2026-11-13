
import argparse
import hashlib
import json
import time

from pathlib import Path

import pandas as pd

from cleaning.dataset import DatasetManager, ErrorProfiler
from cleaning.task import CleaningTask
from cleaning.llm import LLMClient
from cleaning.cleaners import (
    DirectLLMCleaner,
    OperatorSynthesizer,
    GuidedOperatorSynthesizer
)
from cleaning.operator import (
    OperatorExecutor,
    OperatorValidator
)
from cleaning.evaluation import AccuracyEvaluator, matching
from cleaning.comparison import OperatorComparison


def parse_eval_rows(value):
    if str(value).lower() == "all":
        return None

    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError(
            "Use a positive integer or 'all'."
        ) from exc

    if count < 1:
        raise argparse.ArgumentTypeError(
            "Evaluation rows must be positive."
        )

    return count


def cli():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--task",
        choices=CleaningTask.all_names(),
        default="FUS"
    )

    parser.add_argument(
        "--dataset",
        choices=["beers", "hospital"],
        default=None
    )

    parser.add_argument(
        "--data",
        type=Path,
        default=Path("data")
    )

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("results")
    )

    parser.add_argument(
        "--mode",
        choices=[
            "direct",
            "synthesized",
            "guided",
            "all"
        ],
        default="all"
    )

    parser.add_argument(
        "--examples",
        type=int,
        default=15
    )

    parser.add_argument(
        "--eval-rows",
        type=parse_eval_rows,
        default=100
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    parser.add_argument(
        "--affected-only",
        action="store_true"
    )

    parser.add_argument(
        "--run-generated",
        action="store_true"
    )

    parser.add_argument(
        "--regenerate",
        action="store_true"
    )

    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help=(
            "Discard previous Direct LLM "
            "checkpoint and run again."
        )
    )

    parser.add_argument(
        "--compare-all",
        action="store_true"
    )

    parser.add_argument(
        "--repairs", type=int, default=1,
        help="Maximum code repair attempts using held-out validation feedback (default: 1)."
    )
    parser.add_argument(
        "--validation-rows", type=int, default=5,
        help="Reserve this many of --examples for validation feedback (default: 5)."
    )
    parser.add_argument("--auto-request", type=str, default=None,
                        help="Natural-language dataset and cleaning requirements. Enables automated pipeline.")
    parser.add_argument("--input-csv", type=Path, default=None,
                        help="Dirty input CSV (optional if one CSV path appears in --auto-request).")
    parser.add_argument("--reference-csv", type=Path, default=None,
                        help="Optional clean reference CSV for train/validation/test evaluation.")
    parser.add_argument("--plan-only", action="store_true",
                        help="Inspect data and select operators without generating or running code.")
    parser.add_argument("--auto-name", type=str, default=None,
                        help="Result folder name for an automated dataset (defaults to parent of dirty.csv).")
    parser.add_argument("--save-cleaned", action="store_true",
                        help="Optionally save one full cleaned CSV per method in automated mode.")
    return parser.parse_args()


def save_outputs(
        out,
        rows,
        metadata,
        evaluator
):
    if not rows:
        raise ValueError(
            "No predictions produced."
        )

    predictions = pd.DataFrame(rows)

    summary = evaluator.summary(
        predictions,
        metadata
    )


    previous_summary = out / "comparison.csv"
    previous_predictions = out / "predictions.csv"
    if previous_summary.is_file() and previous_predictions.is_file():
        old_summary = pd.read_csv(previous_summary)
        old_predictions = pd.read_csv(previous_predictions)
        run_id = summary["experiment_id"].iloc[0]
        if "experiment_id" in old_summary and old_summary["experiment_id"].eq(run_id).all():
            replaced = set(summary["method"])
            summary = pd.concat([
                old_summary[~old_summary["method"].isin(replaced)], summary
            ], ignore_index=True)
            predictions = pd.concat([
                old_predictions[~old_predictions["method"].isin(replaced)], predictions
            ], ignore_index=True)
        else:

            old_summary.to_csv(out / "previous_comparison.csv", index=False)
            old_predictions.to_csv(out / "previous_predictions.csv", index=False)

    predictions.to_csv(
        out / "predictions.csv",
        index=False
    )

    summary.to_csv(
        out / "comparison.csv",
        index=False
    )

    print(
        "\n=== EXPERIMENT RESULTS ==="
    )

    print(
        summary.to_string(index=False)
    )

    print(
        f"\nResults saved in: {out}"
    )


def compare_all(results_root):
    comparator = OperatorComparison(
        results_root=results_root
    )

    comparison, csv_path = (
        comparator.build()
    )

    chart_path = (
        comparator.plot_all_charts(
            comparison
        )
    )

    display_columns = [
        "operator",
        "dataset",
        "error_correction_accuracy_direct_llm",
        "error_correction_accuracy_synthesized_operator",
        "error_correction_accuracy_guided_operator",
        "correction_difference_pp",
        "winner"
    ]

    print(
        "\n=== ERROR CORRECTION "
        "ACCURACY COMPARISON ==="
    )

    print(
        comparison[
            display_columns
        ].to_string(index=False)
    )

    print(
        f"\nCombined results: {csv_path}"
    )

    print(
        f"Chart: {chart_path}"
    )



def run_operator(args, task, demos, validation, evaluation, out, method,
                 synthesizer_type, code_path, get_client, schema):

    started = time.perf_counter()
    client = None
    call_start = 0
    synthesizer = None
    history = []
    previous_source = None
    feedback = None
    selected_source = None
    selected_check = None
    first_source = None
    initial_validation_score = None
    last_error = "No executable operator produced"
    validation_seconds = 0.0
    generation_seconds = 0.0
    repair_attempts = 0
    existing = code_path.is_file() and bool(code_path.read_text(encoding="utf-8").strip())
    origin = "existing" if existing and not args.regenerate else "new"
    if origin == "existing" and args.repairs:
        print(f"[Warning] {code_path}: cached code may have seen validation rows. "
              "Use --regenerate for an unbiased repair comparison.")

    for attempt in range(args.repairs + 1):
        attempt_started = time.perf_counter()
        calls_before = None
        source = None
        entry = {"attempt": attempt, "source": None, "syntax_valid": False,
                 "execution_success": False, "validation_accuracy": None, "feedback": []}
        try:
            if attempt == 0 and origin == "existing":
                source = code_path.read_text(encoding="utf-8")
            else:
                if synthesizer is None:
                    client = get_client()
                    call_start = len(client.calls)
                    synthesizer = synthesizer_type(client, task, schema=schema)
                synthesizer.last_source = None
                calls_before = len(client.calls)
                generation_started = time.perf_counter()
                try:
                    source = synthesizer.generate(demos, previous_source, feedback)
                finally:
                    generation_seconds += time.perf_counter() - generation_started
            entry["source"] = source
            if attempt == 0:
                first_source = source
            OperatorValidator.validate(source)
            if synthesizer_type is GuidedOperatorSynthesizer:
                synthesizer_type(None, task, schema=schema).validate_operator(source)
            entry["syntax_valid"] = True
            if not args.run_generated:

                code_path.write_text(source, encoding="utf-8")
                selected_source = source
                history.append(entry)
                break

            candidate_path = out / (code_path.stem + "_candidate.py")
            candidate_path.write_text(source, encoding="utf-8")
            check_started = time.perf_counter()
            try:
                values, errors = OperatorExecutor(candidate_path).run(
                    validation, task, allowed=True)
            finally:
                validation_seconds += time.perf_counter() - check_started
            correct = sum(
                row.row_id not in errors and
                matching(values.get(row.row_id), row.clean, task)
                for row in validation
            )
            score = correct / len(validation)
            if initial_validation_score is None:
                initial_validation_score = score
            entry["execution_success"] = not errors
            entry["validation_accuracy"] = score
            entry["execution_errors"] = errors
            feedback = [
                {"input": row.dirty, "expected": row.clean,
                 "actual": values.get(row.row_id),
                 "error": errors.get(row.row_id, "incorrect cleaning")}
                for row in validation
                if row.row_id in errors or
                   not matching(values.get(row.row_id), row.clean, task)
            ][:8]
            entry["feedback"] = feedback
            check = (correct, -len(errors))
            if selected_check is None or check > selected_check:
                selected_source, selected_check = source, check
            previous_source = source
            history.append(entry)
            if not feedback:
                break
        except Exception as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            previous_source = source or (synthesizer.last_source if synthesizer else None)
            feedback = [{"error": last_error}]
            entry.update(source=previous_source, error=last_error, feedback=feedback)
            history.append(entry)
        finally:
            attempt_calls = client.calls[calls_before:] if calls_before is not None else []
            known = all(c.get("usage_available", True) for c in attempt_calls)
            entry.update(
                seconds=round(time.perf_counter() - attempt_started, 3),
                api_calls=len(attempt_calls),
                input_tokens=sum(c.get("input_tokens", 0) for c in attempt_calls) if known else None,
                output_tokens=sum(c.get("output_tokens", 0) for c in attempt_calls) if known else None,
            )
            (out / (code_path.stem + "_candidate.py")).unlink(missing_ok=True)
        print(f"{method}: attempt {attempt}, validation feedback items: {len(feedback or [])}")
        if not args.run_generated or attempt == args.repairs:
            break
        repair_attempts += 1

    history_path = out / "debug" / (code_path.stem + "_history.json")
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps(history, ensure_ascii=False, indent=2), encoding="utf-8")
    if not args.run_generated:
        if selected_source is None:
            raise RuntimeError(f"{method}: {last_error}")
        print(f"Source saved: {code_path}. Review before using --run-generated.")
        return [], {}

    execution_started = time.perf_counter()
    if selected_source is not None:
        code_path.write_text(selected_source, encoding="utf-8")
        values, errors = OperatorExecutor(code_path).run(evaluation, task, allowed=True)
    else:
        values = {row.row_id: None for row in evaluation}
        errors = {row.row_id: last_error for row in evaluation}
    execution_seconds = time.perf_counter() - execution_started


    initial_path = out / "debug" / (code_path.stem + "_initial.py")
    initial_started = time.perf_counter()
    if first_source is not None:
        initial_path.parent.mkdir(parents=True, exist_ok=True)
        initial_path.write_text(first_source, encoding="utf-8")
        if first_source == selected_source:
            initial_values, initial_errors = values, errors
        else:
            try:
                initial_values, initial_errors = OperatorExecutor(initial_path).run(
                    evaluation, task, allowed=True)
            except Exception as exc:
                initial_values = {row.row_id: None for row in evaluation}
                initial_errors = {row.row_id: str(exc) for row in evaluation}
    else:
        initial_values = {row.row_id: None for row in evaluation}
        initial_errors = {row.row_id: "Initial generation failed" for row in evaluation}
    initial_evaluation_seconds = time.perf_counter() - initial_started

    evaluator = AccuracyEvaluator()
    initial_rows = evaluator.rows(method, task, evaluation, initial_values, initial_errors)
    initial_summary = evaluator.summary(pd.DataFrame(initial_rows), {}).iloc[0]
    calls = client.calls[call_start:] if client is not None else []
    usage_known = all(call.get("usage_available", True) for call in calls)
    metadata = {
        "elapsed_seconds": round(time.perf_counter() - started - initial_evaluation_seconds, 3),
        "generation_seconds": round(generation_seconds, 3),
        "execution_seconds": round(execution_seconds, 3),
        "demonstration_seconds": round(validation_seconds, 3),
        "validation_seconds": round(validation_seconds, 3),
        "validation_accuracy": selected_check[0] / len(validation) if selected_check else None,
        "initial_validation_accuracy": initial_validation_score,
        "initial_evaluation_seconds": round(initial_evaluation_seconds, 3),
        "api_calls": len(calls),
        "input_tokens": sum(c.get("input_tokens", 0) for c in calls) if usage_known else None,
        "output_tokens": sum(c.get("output_tokens", 0) for c in calls) if usage_known else None,
        "operator_origin": origin,
        "syntax_valid": selected_source is not None,
        "repair_attempts": repair_attempts,
        "attempt_success_rate": sum(h["execution_success"] for h in history) / len(history),
        "operator_success": selected_source is not None and not errors,
        "execution_success_rate": (len(evaluation) - len(errors)) / len(evaluation),
        "initial_accuracy": initial_summary["accuracy"],
        "initial_error_correction_accuracy": initial_summary["error_correction_accuracy"],
    }
    final_rows = evaluator.rows(method, task, evaluation, values, errors)
    eval_path = out / "debug" / (code_path.stem + "_evaluation.json")
    eval_path.write_text(json.dumps({"initial": initial_rows, "selected": final_rows},
                                    ensure_ascii=False, indent=2), encoding="utf-8")
    return final_rows, metadata


def main():
    args = cli()

    if args.compare_all:
        compare_all(args.out)
        return

    if args.auto_request:
        from cleaning.auto_pipeline import run_auto
        run_auto(args)
        return


    if (
            args.examples < 1
            or args.batch_size < 1
            or args.repairs < 0
            or not 0 < args.validation_rows < args.examples
    ):
        raise ValueError(
            "Examples and batch size "
            "must be positive, repairs non-negative, and 0 < validation-rows < examples."
        )

    task = CleaningTask.create(
        name=args.task,
        dataset=args.dataset
    )

    dataset = DatasetManager(
        args.data,
        task
    )

    print(
        "\n=== DATASET PROFILE ==="
    )

    print(
        ErrorProfiler.analyze(
            dataset,
            task
        )
    )

    demos, evaluation = dataset.split(
        demonstration_count=args.examples,
        evaluation_count=args.eval_rows,
        seed=args.seed,
        affected_only=args.affected_only
    )


    validation = demos[-args.validation_rows:]
    demos = demos[:-args.validation_rows]

    print(
        f"Dataset: {task.dataset}; "
        f"Task: {task.name}; "
        f"Generation examples: {len(demos)}; "
        f"Validation: {len(validation)}; Evaluation: {len(evaluation)}"
    )

    if task.name == "FUS":
        out = (
                args.out
                / "FUS"
                / task.dataset
        )
    else:
        out = (
                args.out
                / task.result_code
        )

    out.mkdir(
        parents=True,
        exist_ok=True
    )

    code_path = (
            out / "generated_operator.py"
    )

    schema = {column: str(dtype) for column, dtype in dataset.dirty.dtypes.items()}
    experiment_id = hashlib.sha256(json.dumps({
        "task": task.__dict__, "schema": schema,
        "demonstrations": [row.__dict__ for row in demos],
        "validation": [row.__dict__ for row in validation],
        "evaluation": [row.__dict__ for row in evaluation],
    }, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
    evaluator = AccuracyEvaluator()

    all_predictions = []
    metadata = {}

    client = None

    def get_client():
        nonlocal client

        if client is None:
            client = LLMClient()

        return client

    # =======================================
    # DIRECT LLM
    # =======================================

    if args.mode in ("direct", "all"):

        print(
            "\nRunning Direct LLM Cleaning..."
        )

        llm = get_client()

        cleaner = DirectLLMCleaner(
            llm,
            task,
            batch_size=args.batch_size,
            checkpoint_path=(
                    out / "direct_checkpoint.json"
            ),
            reset_checkpoint=(
                args.reset_checkpoint
            ),
            schema=schema
        )


        values, errors = cleaner.run(
            demos,
            evaluation
        )

        all_predictions.extend(
            evaluator.rows(
                "direct_llm",
                task,
                evaluation,
                values,
                errors
            )
        )

        metadata["direct_llm"] = {
            "elapsed_seconds": round(
                cleaner.stats[
                    "elapsed_seconds"
                ],
                3
            ),
            "api_calls": cleaner.stats[
                "api_calls"
            ],
            "input_tokens": cleaner.stats[
                "input_tokens"
            ],
            "output_tokens": cleaner.stats[
                "output_tokens"
            ],
            "operator_origin": "N/A",
            "syntax_valid": "N/A",
            "validation_accuracy": "N/A"
        }



    for mode, method, synthesizer_type, code_path in (
            ("synthesized", "synthesized_operator", OperatorSynthesizer, out / "generated_operator.py"),
            ("guided", "guided_operator", GuidedOperatorSynthesizer, out / "generated_guided_operator.py"),
    ):
        if args.mode not in (mode, "all"):
            continue
        print(f"\nPreparing {method}...")
        rows, details = run_operator(
            args, task, demos, validation, evaluation, out, method, synthesizer_type,
            code_path, get_client, schema)
        all_predictions.extend(rows)
        metadata[method] = details


    if not all_predictions:
        print("Operator sources saved. Use --run-generated to evaluate them.")
        return

    for details in metadata.values():
        details["experiment_id"] = experiment_id
    save_outputs(
        out,
        all_predictions,
        metadata,
        evaluator
    )


if __name__ == "__main__":
    main()
