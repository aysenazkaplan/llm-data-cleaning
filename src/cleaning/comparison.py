
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import pandas as pd

from cleaning.task import CleaningTask


class OperatorComparison:

    def __init__(self, results_root="results"):
        self.results_root = Path(results_root)

    @staticmethod
    def _number(value):
        value = pd.to_numeric(value, errors="coerce")
        return float(value) if pd.notna(value) else float("nan")

    @staticmethod
    def _integer(value):
        value = pd.to_numeric(value, errors="coerce")
        return int(value) if pd.notna(value) else float("nan")

    def _experiments(self):
        experiments = []

        for name in CleaningTask.all_names():
            for dataset in CleaningTask.available_datasets(name):

                task = CleaningTask.create(
                    name=name,
                    dataset=dataset
                )

                if task.name == "FUS":
                    folder = (
                            self.results_root
                            / "FUS"
                            / task.dataset
                    )
                else:
                    folder = (
                            self.results_root
                            / task.result_code
                    )

                experiments.append((task, folder))

        return experiments

    def _read_experiment(self, task, folder):
        path = folder / "comparison.csv"

        if not path.is_file():
            print(f"Missing results: {path}")
            return []

        frame = pd.read_csv(path)

        required = {
            "method",
            "accuracy",
            "evaluated_rows"
        }

        if not required.issubset(frame.columns):
            print(f"Invalid results: {path}")
            return []

        records = []

        for _, row in frame.iterrows():
            method = row["method"]

            if method not in {
                "direct_llm",
                "synthesized_operator",
                "guided_operator"
            }:
                continue

            records.append({
                "operator": task.name,
                "dataset": task.dataset,
                "column": ", ".join(task.columns),
                "method": method,

                "evaluated_rows": self._integer(
                    row.get("evaluated_rows")
                ),

                "correct_rows": self._integer(
                    row.get("correct_rows")
                ),

                "accuracy": self._number(
                    row.get("accuracy")
                ),

                "affected_rows": self._integer(
                    row.get("affected_rows")
                ),

                "corrected_rows": self._integer(
                    row.get("corrected_rows")
                ),

                "error_correction_accuracy": self._number(
                    row.get("error_correction_accuracy")
                ),

                "preservation_accuracy": self._number(
                    row.get("preservation_accuracy")
                ),

                "failed_rows": self._integer(
                    row.get("failed_rows")
                ),

                "elapsed_seconds": self._number(
                    row.get("elapsed_seconds")
                ),

                "api_calls": self._integer(
                    row.get("api_calls")
                ),

                "input_tokens": self._integer(
                    row.get("input_tokens")
                ),

                "output_tokens": self._integer(
                    row.get("output_tokens")
                ),
                **{metric: self._number(row.get(metric)) for metric in (
                    "generation_seconds", "execution_seconds", "demonstration_seconds",
                    "repair_attempts", "attempt_success_rate", "execution_success_rate",
                    "initial_accuracy", "initial_error_correction_accuracy"
                )}
            })

        return records

    def build(self):
        records = []

        for task, folder in self._experiments():
            records.extend(
                self._read_experiment(task, folder)
            )

        if not records:
            raise ValueError(
                "No experiment results found."
            )

        frame = pd.DataFrame(records)

        keys = [
            "operator",
            "dataset",
            "column",
            "method"
        ]

        if frame.duplicated(keys).any():
            raise ValueError(
                "Duplicate experiment results found."
            )

        metrics = [
            "evaluated_rows",
            "correct_rows",
            "accuracy",
            "affected_rows",
            "corrected_rows",
            "error_correction_accuracy",
            "preservation_accuracy",
            "failed_rows",
            "elapsed_seconds",
            "api_calls",
            "input_tokens",
            "output_tokens",
            "generation_seconds", "execution_seconds", "demonstration_seconds",
            "repair_attempts", "attempt_success_rate", "execution_success_rate",
            "initial_accuracy", "initial_error_correction_accuracy"
        ]

        comparison = frame.pivot(
            index=["operator", "dataset", "column"],
            columns="method",
            values=metrics
        )

        comparison.columns = [
            f"{metric}_{method}"
            for metric, method in comparison.columns
        ]

        comparison = comparison.reset_index()

        required_columns = [
            "accuracy_direct_llm",
            "accuracy_synthesized_operator",
            "error_correction_accuracy_direct_llm",
            "error_correction_accuracy_synthesized_operator",
            "evaluated_rows_direct_llm",
            "evaluated_rows_synthesized_operator",
            "affected_rows_direct_llm",
            "affected_rows_synthesized_operator",
            "error_correction_accuracy_guided_operator",
            "accuracy_guided_operator",
            "evaluated_rows_guided_operator",
            "affected_rows_guided_operator"
        ]

        for column in required_columns:
            if column not in comparison.columns:
                comparison[column] = float("nan")

        # Overall accuracy difference
        comparison["accuracy_difference_pp"] = (
                (
                        comparison["accuracy_synthesized_operator"]
                        - comparison["accuracy_direct_llm"]
                ) * 100
        ).round(2)

        # Error correction accuracy difference
        comparison["correction_difference_pp"] = (
                (
                        comparison[
                            "error_correction_accuracy_synthesized_operator"
                        ]
                        - comparison[
                            "error_correction_accuracy_direct_llm"
                        ]
                ) * 100
        ).round(2)


        def determine_winner(row):
            methods = ("direct_llm", "synthesized_operator", "guided_operator")
            scores = {m: row.get("error_correction_accuracy_" + m) for m in methods}
            if any(pd.isna(v) for v in scores.values()):
                return "Incomplete"
            sizes = {(row.get("evaluated_rows_" + m), row.get("affected_rows_" + m)) for m in methods}
            if len(sizes) != 1:
                return "Different evaluation sizes"
            best = max(scores.values())
            winners = [m for m, value in scores.items() if value == best]
            return winners[0] if len(winners) == 1 else "Tie"

        comparison["winner"] = comparison.apply(
            determine_winner,
            axis=1
        )

        order = {
            name: pos
            for pos, name in enumerate(
                CleaningTask.all_names()
            )
        }

        comparison["_order"] = (
            comparison["operator"].map(order)
        )

        comparison = comparison.sort_values(
            ["_order", "dataset"]
        ).drop(columns="_order")

        self.results_root.mkdir(
            parents=True,
            exist_ok=True
        )

        output_path = (
                self.results_root
                / "all_operators_comparison.csv"
        )

        comparison.to_csv(
            output_path,
            index=False
        )

        return comparison, output_path

    def plot_accuracy_chart(self, comparison):


        labels = [
            f"{row.operator}\n{row.dataset}"
            for row in comparison.itertuples()
        ]

        direct_col = (
            "error_correction_accuracy_direct_llm"
        )

        synth_col = (
            "error_correction_accuracy_synthesized_operator"
        )

        positions = list(range(len(labels)))
        width = 0.25
        fig, ax = plt.subplots(figsize=(11, 6))
        for offset, method, label in (
                (-1, "direct_llm", "Direct LLM"),
                (0, "synthesized_operator", "Synthesized Operator"),
                (1, "guided_operator", "Guided Operator"),
        ):
            values = pd.to_numeric(comparison["error_correction_accuracy_" + method], errors="coerce")
            bars = ax.bar([x + offset * width for x in positions], values, width, label=label)
            for bar in bars:
                height = bar.get_height()
                if pd.notna(height) and height >= 0:
                    ax.annotate(f"{height * 100:.1f}%", (bar.get_x() + bar.get_width()/2, height),
                                xytext=(0, 4), textcoords="offset points", ha="center", fontsize=9)

        ax.set_xticks(positions)
        ax.set_xticklabels(labels)

        ax.set_ylim(0, 1.12)

        ax.set_xlabel("Cleaning Operator / Dataset")
        ax.set_ylabel("Error Correction Accuracy")

        ax.set_title(
            "Error Correction Accuracy Across Operators"
        )

        ax.legend()
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)

        fig.tight_layout()

        output_path = (
                self.results_root
                / "figures"
                / "all_operators_error_accuracy.png"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)

        fig.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close(fig)

        return output_path


    @staticmethod
    def _method_specs(generated_only=False):
        specs = [
            ("direct_llm", "Direct LLM", "#4C78A8"),
            ("synthesized_operator", "Synthesized (Unrestricted)", "#F58518"),
            ("guided_operator", "Guided (Template)", "#54A24B"),
        ]
        return specs[1:] if generated_only else specs

    @staticmethod
    def _axis_labels(comparison):
        return [f"{op}\n{ds}" for op, ds in zip(comparison["operator"], comparison["dataset"])]

    def _plot_grouped(self, comparison, values_by_method, title, ylabel,
                      filename, percentage=False, generated_only=False,
                      log_scale=False):
        import numpy as np

        specs = self._method_specs(generated_only)
        x = np.arange(len(comparison))
        width = 0.78 / len(specs)
        fig, ax = plt.subplots(figsize=(max(10, len(comparison) * 1.65), 6))
        plotted_values = []

        for idx, (method, label, color) in enumerate(specs):
            values = pd.to_numeric(values_by_method.get(method, pd.Series(float("nan"), index=comparison.index)), errors="coerce")
            values = values.astype(float)
            if percentage:
                values = values * 100
            plotted_values.extend(values[np.isfinite(values)].tolist())
            offsets = x + (idx - (len(specs) - 1) / 2) * width
            bars = ax.bar(offsets, values, width, label=label, color=color)
            for bar, val in zip(bars, values):
                if pd.isna(val) or not np.isfinite(val):
                    continue
                label_text = f"{val:.1f}%" if percentage else (f"{val:,.0f}" if val >= 100 else f"{val:.2f}")
                ax.annotate(label_text, (bar.get_x() + bar.get_width() / 2, val),
                            xytext=(0, 3), textcoords="offset points",
                            ha="center", va="bottom", fontsize=7, rotation=0)

        ax.set_xticks(x)
        ax.set_xticklabels(self._axis_labels(comparison))
        ax.set_title(title)
        ax.set_xlabel("Operator / Dataset")
        ax.set_ylabel(ylabel)
        if percentage:
            ax.set_ylim(0, max(110, max(plotted_values, default=0) * 1.13))
        elif log_scale and plotted_values and min(plotted_values) > 0:
            ax.set_yscale("log")
            ax.set_ylim(top=max(plotted_values) * 2.0)
        elif plotted_values:
            ax.set_ylim(0, max(plotted_values) * 1.18 if max(plotted_values) > 0 else 1)
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=len(specs), frameon=False)
        fig.tight_layout()
        output_path = self.results_root / "figures" / filename
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        return output_path

    @staticmethod
    def _metric_values(comparison, metric, generated_only=False):
        return {
            method: pd.to_numeric(comparison.get(f"{metric}_{method}", pd.Series(float("nan"), index=comparison.index)), errors="coerce")
            for method, _, _ in OperatorComparison._method_specs(generated_only)
        }

    def plot_overall_accuracy_chart(self, comparison):
        return self._plot_grouped(
            comparison, self._metric_values(comparison, "accuracy"),
            "Overall Cleaning Accuracy", "Accuracy (%)",
            "all_operators_overall_accuracy.png", percentage=True)

    def plot_token_consumption_chart(self, comparison):
        values = {}
        for method, _, _ in self._method_specs():
            input_col = f"input_tokens_{method}"
            output_col = f"output_tokens_{method}"
            input_values = pd.to_numeric(comparison.get(input_col, pd.Series(float("nan"), index=comparison.index)), errors="coerce")
            output_values = pd.to_numeric(comparison.get(output_col, pd.Series(float("nan"), index=comparison.index)), errors="coerce")
            values[method] = input_values + output_values
        return self._plot_grouped(
            comparison, values, "Total Token Consumption (Input + Output)",
            "Tokens", "all_operators_tokens.png")

    def plot_elapsed_time_chart(self, comparison):
        return self._plot_grouped(
            comparison, self._metric_values(comparison, "elapsed_seconds"),
            "End-to-End Runtime", "Elapsed time (seconds)",
            "all_operators_elapsed_time.png")

    def plot_generation_time_chart(self, comparison):
        return self._plot_grouped(
            comparison, self._metric_values(comparison, "generation_seconds", True),
            "Code Generation Time", "Generation time (seconds)",
            "all_operators_generation_time.png", generated_only=True)

    def plot_execution_time_chart(self, comparison):
        return self._plot_grouped(
            comparison, self._metric_values(comparison, "execution_seconds", True),
            "Generated Operator Execution Time", "Execution time (seconds)",
            "all_operators_execution_time.png", generated_only=True)

    def plot_attempt_success_rate_chart(self, comparison):
        return self._plot_grouped(
            comparison, self._metric_values(comparison, "attempt_success_rate", True),
            "Operator Generation Success Rate", "Success rate (%)",
            "all_operators_generation_success.png", percentage=True, generated_only=True)

    def plot_execution_success_rate_chart(self, comparison):
        return self._plot_grouped(
            comparison, self._metric_values(comparison, "execution_success_rate", True),
            "Generated Operator Execution Success Rate", "Success rate (%)",
            "all_operators_execution_success.png", percentage=True, generated_only=True)

    def plot_all_charts(self, comparison):
        """Create only Xiao's four requested evaluation figures."""
        methods = (
            self.plot_accuracy_chart,
            self.plot_token_consumption_chart,
            self.plot_elapsed_time_chart,
            self.plot_attempt_success_rate_chart,
        )
        paths = [plot(comparison) for plot in methods]
        self.organize_results()
        return paths

    @staticmethod
    def _move_without_overwriting(source, destination):

        destination.parent.mkdir(parents=True, exist_ok=True)
        candidate = destination
        index = 1
        while candidate.exists():
            candidate = destination.with_name(
                f"{destination.stem}_{index}{destination.suffix}"
            )
            index += 1
        shutil.move(str(source), str(candidate))
        return candidate

    def organize_results(self):

        figures = self.results_root / "figures"
        figures.mkdir(parents=True, exist_ok=True)
        moved = []


        obsolete_charts = (
            "all_operators_overall_accuracy.png",
            "all_operators_generation_time.png",
            "all_operators_execution_time.png",
            "all_operators_execution_success.png",
        )
        for name in obsolete_charts:
            for obsolete in (figures / name, self.results_root / name):
                if obsolete.is_file():
                    obsolete.unlink()


        for path in self.results_root.glob("all_operators_*.png"):
            moved.append(self._move_without_overwriting(path, figures / path.name))


        for _, folder in self._experiments():
            if not folder.is_dir():
                continue
            debug = folder / "debug"
            patterns = (
                "direct_checkpoint.json",
                "generated_operator_initial.py",
                "generated_operator_evaluation.json",
                "generated_operator_history.json",
                "generated_guided_operator_initial.py",
                "generated_guided_operator_evaluation.json",
                "generated_guided_operator_history.json",
                "previous_comparison.csv",
                "previous_predictions.csv",
            )
            for filename in patterns:
                path = folder / filename
                if path.is_file():
                    moved.append(self._move_without_overwriting(path, debug / filename))
        return moved
