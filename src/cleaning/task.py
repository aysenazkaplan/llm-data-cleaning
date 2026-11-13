
from dataclasses import dataclass


@dataclass(frozen=True)
class CleaningTask:
    name: str
    dataset: str
    columns: tuple[str, ...]
    instruction: str
    numeric: bool = False

    @property
    def result_code(self):
        return self.name

    @classmethod
    def all_names(cls):
        return ["FUS", "CFC", "NFN", "CC"]

    @classmethod
    def available_datasets(cls, name):
        if name == "FUS":
            return ["beers", "hospital"]
        if name in ("CFC", "NFN"):
            return ["beers"]
        if name == "CC":
            return ["hospital"]

        raise ValueError(f"Unknown task: {name}")

    @classmethod
    def create(cls, name, dataset=None):

        available = cls.available_datasets(name)

        if dataset is None:
            dataset = (
                "hospital" if name == "CC" else "beers"
            )

        if dataset not in available:
            raise ValueError(
                f"Task {name} does not support "
                f"dataset '{dataset}'. "
                f"Available datasets: {available}"
            )

        if name == "FUS" and dataset == "beers":
            return cls(
                name="FUS",
                dataset="beers",
                columns=("ounces",),
                numeric=True,
                instruction=(
                    "Perform Format and Unit Standardization "
                    "on the 'ounces' column of the Beers dataset. "
                    "Analyze dirty-clean demonstrations to "
                    "identify inconsistent representations. "
                    "Infer generalizable cleaning rules and "
                    "apply them to unseen records. "
                    "Preserve the intended numeric meaning. "
                    "Return a numeric value or null."
                )
            )

        if name == "FUS" and dataset == "hospital":
            return cls(
                name="FUS",
                dataset="hospital",
                columns=("Score",),
                numeric=False,
                instruction=(
                    "Perform Format and Unit Standardization "
                    "on the 'Score' column of the Hospital "
                    "dataset. Study the provided dirty-clean "
                    "demonstrations to identify representation "
                    "inconsistencies and infer generalizable "
                    "cleaning rules. Preserve already-correct "
                    "values, avoid unsupported corrections, "
                    "and return the cleaned value in the same "
                    "canonical string format as the examples. "
                    "Return null only for truly missing values."
                )
            )

        if name == "CFC":
            return cls(
                name="CFC",
                dataset="beers",
                columns=("city", "state"),
                numeric=False,
                instruction=(
                    "Perform semantic data cleaning on the "
                    "'city' and 'state' columns. Analyze "
                    "dirty-clean demonstrations to identify "
                    "inconsistencies within and between fields. "
                    "Infer generalizable cleaning rules. "
                    "Preserve correct values, avoid inventing "
                    "information, and return both fields."
                )
            )

        if name == "NFN":
            return cls(
                name="NFN",
                dataset="beers",
                columns=("abv",),
                numeric=True,
                instruction=(
                    "Perform Numeric Format Normalization on "
                    "the 'abv' column. Infer normalization "
                    "rules from dirty-clean demonstrations "
                    "and apply them to unseen records. "
                    "Preserve numeric meaning and valid values. "
                    "Return a numeric value or null."
                )
            )

        if name == "CC":
            return cls(
                name="CC",
                dataset="hospital",
                columns=("city",),
                numeric=False,
                instruction=(
                    "Perform Categorical Canonicalization on "
                    "the 'city' column of the Hospital dataset. "
                    "Infer canonical representations from "
                    "dirty-clean demonstrations and apply "
                    "generalizable rules to unseen records. "
                    "Preserve correct categories and avoid "
                    "unsupported corrections. "
                    "Return the cleaned city or null."
                )
            )

        raise ValueError(
            f"Unsupported task/dataset: {name}/{dataset}"
        )
