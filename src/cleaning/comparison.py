
from pathlib import Path

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
                "synthesized_operator"
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
                )
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
            "output_tokens"
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
            "affected_rows_synthesized_operator"
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

        # Winner is based on ERROR CORRECTION ACCURACY.
        def determine_winner(row):
            direct = row[
                "error_correction_accuracy_direct_llm"
            ]

            synth = row[
                "error_correction_accuracy_synthesized_operator"
            ]

            direct_count = row[
                "affected_rows_direct_llm"
            ]

            synth_count = row[
                "affected_rows_synthesized_operator"
            ]

            direct_total = row[
                "evaluated_rows_direct_llm"
            ]

            synth_total = row[
                "evaluated_rows_synthesized_operator"
            ]

            if pd.isna(direct) or pd.isna(synth):
                return "Incomplete"

            if (
                    pd.isna(direct_count)
                    or pd.isna(synth_count)
                    or pd.isna(direct_total)
                    or pd.isna(synth_total)
            ):
                return "Incomplete"

            if (
                    direct_count != synth_count
                    or direct_total != synth_total
            ):
                return "Different evaluation sizes"

            if synth > direct:
                return "Synthesized Operator"

            if direct > synth:
                return "Direct LLM"

            return "Tie"

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

        direct = pd.to_numeric(
            comparison[direct_col],
            errors="coerce"
        )

        synth = pd.to_numeric(
            comparison[synth_col],
            errors="coerce"
        )

        positions = list(range(len(labels)))
        width = 0.35

        fig, ax = plt.subplots(
            figsize=(11, 6)
        )

        bars_direct = ax.bar(
            [x - width / 2 for x in positions],
            direct,
            width,
            label="Direct LLM"
        )

        bars_synth = ax.bar(
            [x + width / 2 for x in positions],
            synth,
            width,
            label="Synthesized Operator"
        )

        # Display percentage values above each bar.
        for bars in (bars_direct, bars_synth):
            for bar in bars:
                height = bar.get_height()

                if pd.notna(height) and height >= 0:
                    ax.annotate(
                        f"{height * 100:.1f}%",
                        (
                            bar.get_x()
                            + bar.get_width() / 2,
                            height
                        ),
                        xytext=(0, 4),
                        textcoords="offset points",
                        ha="center",
                        va="bottom",
                        fontsize=9
                    )

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
                / "all_operators_error_accuracy.png"
        )

        fig.savefig(
            output_path,
            dpi=200,
            bbox_inches="tight"
        )

        plt.close(fig)

        return output_path
