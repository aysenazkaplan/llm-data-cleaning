
from dataclasses import dataclass
from pathlib import Path
import random

import pandas as pd


@dataclass(frozen=True)
class Example:
    row_id: int
    dirty: dict
    clean: dict


class DatasetManager:

    BALANCED_TASKS = {"CFC", "CC", "FUS"}

    COLUMN_MAPPINGS = {
        "hospital": {
            "clean": {"City": "city"},
            "dirty": {"City": "city"}
        }
    }

    def __init__(self, data_root, task):
        self.task = task
        self.dataset_name = task.dataset

        folder = Path(data_root) / task.dataset

        self.clean = pd.read_csv(
            folder / "clean.csv",
            dtype=str,
            keep_default_na=False
        )

        self.dirty = pd.read_csv(
            folder / "dirty.csv",
            dtype=str,
            keep_default_na=False
        )

        mappings = self.COLUMN_MAPPINGS.get(
            task.dataset, {}
        )

        self.clean = self.clean.rename(
            columns=mappings.get("clean", {})
        )

        self.dirty = self.dirty.rename(
            columns=mappings.get("dirty", {})
        )

        if len(self.clean) != len(self.dirty):
            raise ValueError(
                "Clean and dirty datasets have "
                "different numbers of rows."
            )

        if task.dataset == "beers":
            self._align_beers()
        else:
            self._align_hospital()

        for column in task.columns:
            if (
                    column not in self.clean.columns
                    or column not in self.dirty.columns
            ):
                raise ValueError(
                    f"Missing target column: {column}"
                )

    def _align_beers(self):
        if (
                "index" not in self.clean.columns
                or "index" not in self.dirty.columns
        ):
            raise ValueError(
                "Beers data requires an index column."
            )

        self.clean = self.clean.set_index(
            "index", verify_integrity=True
        )

        self.dirty = self.dirty.set_index(
            "index", verify_integrity=True
        )

        if set(self.clean.index) != set(
                self.dirty.index
        ):
            raise ValueError(
                "Clean and dirty row IDs differ."
            )

        self.clean = self.clean.sort_index()
        self.dirty = self.dirty.loc[
            self.clean.index
        ]

        if (
                "id" in self.clean.columns
                and "id" in self.dirty.columns
                and not self.clean["id"].equals(
            self.dirty["id"]
        )
        ):
            raise ValueError(
                "Beers row alignment failed."
            )

        self.clean.index = range(len(self.clean))
        self.dirty.index = range(len(self.dirty))

    def _align_hospital(self):

        self.clean = self.clean.reset_index(
            drop=True
        )
        self.dirty = self.dirty.reset_index(
            drop=True
        )

        shared = [
            col for col in self.clean.columns
            if col in self.dirty.columns
               and col not in self.task.columns
        ]

        if not shared:
            raise ValueError(
                "No shared columns available for "
                "Hospital alignment verification."
            )


        checked = 0
        matched = 0

        for col in shared:
            a = self.clean[col]
            b = self.dirty[col]
            checked += len(a)
            matched += int((a == b).sum())

        if checked and matched / checked < 0.8:
            raise ValueError(
                "Hospital row alignment is uncertain. "
                "Check clean and dirty row ordering."
            )

    @staticmethod
    def _cell(value):
        if value is None or pd.isna(value):
            return None

        value = str(value).strip()

        if value == "":
            return None

        return value

    def _row(self, frame, row_id, clean=False):
        result = {}

        for col in self.task.columns:
            value = self._cell(
                frame.at[row_id, col]
            )

            if (
                    clean
                    and self.task.numeric
                    and value is not None
            ):
                try:
                    value = float(value)
                except (TypeError, ValueError):
                    raise ValueError(
                        f"Invalid clean numeric value "
                        f"at row {row_id}, column {col}: "
                        f"{value!r}"
                    )

            result[col] = value

        return result

    @staticmethod
    def _different(dirty, clean, task):
        for col in task.columns:
            a = dirty[col]
            b = clean[col]

            if a is None or b is None:
                if a is not None or b is not None:
                    return True
                continue

            if task.numeric:
                try:
                    if float(a) != float(b):
                        return True
                except (TypeError, ValueError):
                    return True
            elif a != b:
                return True

        return False

    def examples(self, affected_only=False):
        result = []

        for row_id in range(len(self.clean)):
            dirty = self._row(
                self.dirty,
                row_id,
                clean=False
            )

            clean = self._row(
                self.clean,
                row_id,
                clean=True
            )

            affected = self._different(
                dirty, clean, self.task
            )

            if affected_only and not affected:
                continue

            result.append(
                Example(
                    row_id=row_id,
                    dirty=dirty,
                    clean=clean
                )
            )

        return result

    def split(
            self,
            demonstration_count,
            evaluation_count,
            seed,
            affected_only=False
    ):


        examples = self.examples(
            affected_only=affected_only
        )

        if demonstration_count < 1:
            raise ValueError(
                "Need at least one demonstration."
            )

        if demonstration_count >= len(examples):
            raise ValueError(
                "Not enough eligible rows for "
                "demonstrations and evaluation."
            )

        rng = random.Random(seed)
        rng.shuffle(examples)

        use_balanced_sampling = (
                self.task.name in self.BALANCED_TASKS
                and not (
                self.task.name == "FUS"
                and self.task.dataset == "beers"
        )
                and not affected_only
        )

        if use_balanced_sampling:
            affected = [
                row for row in examples
                if self._different(
                    row.dirty, row.clean, self.task
                )
            ]

            unchanged = [
                row for row in examples
                if not self._different(
                    row.dirty, row.clean, self.task
                )
            ]

            n_affected = min(
                len(affected),
                (demonstration_count + 1) // 2
            )

            n_unchanged = (
                    demonstration_count - n_affected
            )

            if len(unchanged) < n_unchanged:
                shortage = (
                        n_unchanged - len(unchanged)
                )

                n_unchanged = len(unchanged)
                n_affected = min(
                    len(affected),
                    n_affected + shortage
                )

            demos = (
                    affected[:n_affected]
                    + unchanged[:n_unchanged]
            )

            if len(demos) != demonstration_count:
                raise ValueError(
                    "Cannot construct requested "
                    "demonstration sample."
                )

            rng.shuffle(demos)

            selected = {
                row.row_id for row in demos
            }

            remaining = [
                row for row in examples
                if row.row_id not in selected
            ]

            rng.shuffle(remaining)

        else:
            demos = examples[:demonstration_count]

            remaining = examples[
                demonstration_count:
            ]

        if evaluation_count is None:
            evaluation = remaining

        else:
            if evaluation_count < 1:
                raise ValueError(
                    "Evaluation count must be positive."
                )

            if evaluation_count > len(remaining):
                raise ValueError(
                    f"Requested {evaluation_count} "
                    f"evaluation rows, but only "
                    f"{len(remaining)} are available. "
                    "Use 'all' to select the remainder."
                )

            evaluation = remaining[
                :evaluation_count
            ]

        if not evaluation:
            raise ValueError(
                "Evaluation set cannot be empty."
            )

        return demos, evaluation


class ErrorProfiler:

    @staticmethod
    def analyze(dataset, task):
        mismatches = {}

        for col in task.columns:
            a = dataset.dirty[col]
            b = dataset.clean[col]

            mismatches[col] = int(
                (~(a.eq(b))).sum()
            )

        return {
            "total_rows": len(dataset.clean),
            "task": task.name,
            "dataset": task.dataset,
            "mismatches_by_column": mismatches,
            "affected_rows": len(
                dataset.examples(affected_only=True)
            )
        }
