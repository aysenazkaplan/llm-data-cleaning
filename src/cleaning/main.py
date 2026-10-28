
import argparse
import time

from pathlib import Path

import pandas as pd

from cleaning.dataset import DatasetManager, ErrorProfiler
from cleaning.task import CleaningTask
from cleaning.llm import LLMClient
from cleaning.cleaners import (
    DirectLLMCleaner,
    OperatorSynthesizer
)
from cleaning.operator import (
    OperatorExecutor,
    OperatorValidator
)
from cleaning.evaluation import AccuracyEvaluator
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
            "synthesize",
            "both"
        ],
        default="both"
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
        comparator.plot_accuracy_chart(
            comparison
        )
    )

    display_columns = [
        "operator",
        "dataset",
        "error_correction_accuracy_direct_llm",
        "error_correction_accuracy_synthesized_operator",
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


def main():
    args = cli()

    if args.compare_all:
        compare_all(args.out)
        return

    if (
            args.examples < 1
            or args.batch_size < 1
    ):
        raise ValueError(
            "Examples and batch size "
            "must be positive."
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

    print(
        f"Dataset: {task.dataset}; "
        f"Task: {task.name}; "
        f"Demonstrations: {len(demos)}; "
        f"Evaluation: {len(evaluation)}"
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

    if args.mode in ("direct", "both"):

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
            )
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



    if args.mode in (
            "synthesize",
            "both"
    ):
        print(
            "\nPreparing Synthesized Operator..."
        )

        existing = (
                code_path.is_file()
                and bool(
            code_path.read_text(
                encoding="utf-8"
            ).strip()
        )
        )

        start = time.perf_counter()
        calls = []

        if existing and not args.regenerate:
            origin = "existing"

            source = code_path.read_text(
                encoding="utf-8"
            )

            print(
                f"Using existing operator: "
                f"{code_path}"
            )

        else:
            origin = "new"

            llm = get_client()
            call_start = len(
                llm.calls
            )

            synthesizer = OperatorSynthesizer(
                llm,
                task
            )

            source = synthesizer.generate(
                demos
            )

            calls = llm.calls[
                call_start:
            ]

            code_path.write_text(
                source,
                encoding="utf-8"
            )

            print(
                "Generated operator saved: "
                f"{code_path}"
            )

        syntax_valid = False
        validation_accuracy = None
        values = {}
        errors = {}

        try:
            OperatorValidator.validate(
                source
            )

            syntax_valid = True

            print(
                "Syntax validation: PASS"
            )

            if not args.run_generated:
                print(
                    f"Review {code_path} "
                    "before execution."
                )

                if all_predictions:
                    save_outputs(
                        out,
                        all_predictions,
                        metadata,
                        evaluator
                    )

                return

            check = (
                OperatorValidator.validate_examples(
                    code_path,
                    demos,
                    task,
                    allowed=True
                )
            )

            validation_accuracy = (
                check["accuracy"]
            )

            print(
                "Demonstration accuracy: "
                f"{validation_accuracy:.4f}"
            )

            executor = OperatorExecutor(
                code_path
            )

            values, errors = executor.run(
                evaluation,
                task,
                allowed=True
            )

        except Exception as exc:
            reason = (
                f"{type(exc).__name__}: {exc}"
            )

            print(
                "Operator failed:",
                reason
            )

            values = {
                row.row_id: None
                for row in evaluation
            }

            errors = {
                row.row_id: reason
                for row in evaluation
            }

        all_predictions.extend(
            evaluator.rows(
                "synthesized_operator",
                task,
                evaluation,
                values,
                errors
            )
        )

        metadata[
            "synthesized_operator"
        ] = {
            "elapsed_seconds": round(
                time.perf_counter() - start,
                3
            ),
            "api_calls": len(calls),
            "input_tokens": sum(
                c.get("input_tokens", 0)
                for c in calls
            ),
            "output_tokens": sum(
                c.get("output_tokens", 0)
                for c in calls
            ),
            "operator_origin": origin,
            "syntax_valid": syntax_valid,
            "validation_accuracy": (
                validation_accuracy
            )
        }


    save_outputs(
        out,
        all_predictions,
        metadata,
        evaluator
    )


if __name__ == "__main__":
    main()
