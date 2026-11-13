
import math

import pandas as pd


def matching(prediction, truth, task):

    if (
            not isinstance(prediction, dict)
            or set(prediction) != set(task.columns)
    ):
        return False

    for col in task.columns:
        predicted = prediction[col]
        expected = truth[col]

        if predicted is None or expected is None:
            if predicted is not None or expected is not None:
                return False
            continue

        if task.numeric:
            try:
                if (
                        not math.isfinite(float(predicted))
                        or not math.isclose(
                    float(predicted),
                    float(expected),
                    rel_tol=0,
                    abs_tol=1e-8
                )
                ):
                    return False
            except (ValueError, TypeError, OverflowError):
                return False
        elif predicted != expected:
            return False

    return True


def is_affected(dirty, clean, task):


    for col in task.columns:
        left = dirty[col]
        right = clean[col]

        if left is None or right is None:
            if left is not None or right is not None:
                return True
            continue

        if task.numeric:
            try:
                if not math.isclose(
                        float(left),
                        float(right),
                        rel_tol=0,
                        abs_tol=1e-8
                ):
                    return True
            except (ValueError, TypeError, OverflowError):
                return True
        elif left != right:
            return True

    return False


class AccuracyEvaluator:

    @staticmethod
    def rows(
            method,
            task,
            evaluation,
            predictions,
            failures
    ):
        records = []

        for row in evaluation:
            prediction = predictions.get(row.row_id)

            records.append({
                "method": method,
                "task": task.name,
                "dataset": task.dataset,
                "row_id": row.row_id,
                "dirty": str(row.dirty),
                "ground_truth": str(row.clean),
                "prediction": str(prediction),
                "affected": is_affected(
                    row.dirty,
                    row.clean,
                    task
                ),
                "correct": matching(
                    prediction,
                    row.clean,
                    task
                ),
                "error": failures.get(row.row_id, "")
            })

        return records

    @staticmethod
    def summary(predictions, metadata):

        results = []

        for method, group in predictions.groupby(
                "method",
                sort=False
        ):
            total = len(group)
            correct = int(group["correct"].sum())

            affected = group[group["affected"]]
            affected_total = len(affected)
            affected_correct = int(
                affected["correct"].sum()
            )

            unchanged = group[~group["affected"]]
            unchanged_total = len(unchanged)
            unchanged_correct = int(
                unchanged["correct"].sum()
            )

            results.append({
                "task": group["task"].iloc[0],
                "dataset": group["dataset"].iloc[0],
                "method": method,
                "evaluated_rows": total,
                "correct_rows": correct,
                "accuracy": round(
                    correct / total, 5
                ),
                "affected_rows": affected_total,
                "corrected_rows": affected_correct,
                "error_correction_accuracy": (
                    round(
                        affected_correct / affected_total,
                        5
                    )
                    if affected_total else float("nan")
                ),
                "unchanged_rows": unchanged_total,
                "preserved_rows": unchanged_correct,
                "preservation_accuracy": (
                    round(
                        unchanged_correct / unchanged_total,
                        5
                    )
                    if unchanged_total else float("nan")
                ),
                "failed_rows": int(
                    group["error"].fillna("").ne("").sum()
                ),
                **metadata.get(method, {})
            })

        return pd.DataFrame(results)
