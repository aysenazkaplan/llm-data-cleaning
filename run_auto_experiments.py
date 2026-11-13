

import subprocess
import sys
import time

from auto_visualization import plot_new_dataset_results
from pathlib import Path


EXPERIMENTS = [
    {
        "name": "hospital",
        "input_csv": "data/hospital/dirty.csv",
        "reference_csv": "data/hospital/clean.csv",
        "request": (
            "Inspect the dataset and identify columns with semantic data quality "
            "issues. Select suitable operators to canonicalize inconsistent "
            "categorical values and standardize inconsistent numeric or "
            "text formats. Preserve valid entries and avoid unsupported corrections."
        ),
        "enabled": True,
        "plan": None,
        "operators": [],
    },
    {
        "name": "beers",
        "input_csv": "data/beers/dirty.csv",
        "reference_csv": "data/beers/clean.csv",
        "request": (
            "Inspect the dataset and identify columns with inconsistent formats, "
            "numeric representations, or conflicting related fields. Select "
            "appropriate cleaning operators and preserve correct values."
        ),
        "enabled": True,
        "plan": None,
        "operators": [],
    },
]

MODE = "all"                   # direct | synthesized | guided | all
PLAN_ONLY = False                # set True to inspect LLM selections
EXAMPLES = 30
VALIDATION_ROWS = 10
EVAL_ROWS = "all"
BATCH_SIZE = 20
SEED = 42
REPAIRS = 2
RUN_GENERATED = True
REGENERATE = True
RESET_CHECKPOINT = False
SAVE_CLEANED = False
OUT = "results"
STOP_ON_ERROR = True
DRY_RUN = False                 # when True, makes no LLM calls
CREATE_FIGURES = True

# change for independent runs
RUN_TAG = ""

PROJECT_ROOT = Path(__file__).resolve().parent
MAIN_FILE = PROJECT_ROOT / "main.py"


def check_settings():
    if MODE not in {"direct", "synthesized", "guided", "all"}:
        raise ValueError(f"Unsupported MODE: {MODE}")
    if not 0 < VALIDATION_ROWS < EXAMPLES:
        raise ValueError("Require 0 < VALIDATION_ROWS < EXAMPLES")
    if REPAIRS < 0 or BATCH_SIZE < 1:
        raise ValueError("REPAIRS must be nonnegative and BATCH_SIZE positive")
    selected = [e for e in EXPERIMENTS if e["enabled"]]
    if len({e["name"] for e in selected}) != len(selected):
        raise ValueError("Each enabled experiment needs a unique name")
    if not MAIN_FILE.is_file():
        raise FileNotFoundError(f"Missing launcher: {MAIN_FILE}")
    for e in selected:
        dirty = PROJECT_ROOT / e["input_csv"]
        if not dirty.is_file():
            raise FileNotFoundError(f"Dirty dataset not found: {dirty}")
        reference = e.get("reference_csv")
        if reference and not (PROJECT_ROOT / reference).is_file():
            raise FileNotFoundError(f"Clean reference not found: {reference}")
        if not str(e.get("request", "")).strip():
            raise ValueError(f"Missing natural-language request for {e['name']}")
    return selected


def build_arguments(experiment):
    name = experiment["name"] + (f"_{RUN_TAG}" if RUN_TAG else "")
    args = [
        "--auto-request", experiment["request"],
        "--input-csv", experiment["input_csv"],
        "--auto-name", name,
        "--out", str(OUT),
        "--mode", MODE,
        "--examples", str(EXAMPLES),
        "--validation-rows", str(VALIDATION_ROWS),
        "--eval-rows", str(EVAL_ROWS),
        "--batch-size", str(BATCH_SIZE),
        "--seed", str(SEED),
        "--repairs", str(REPAIRS),
    ]
    if experiment.get("reference_csv"):
        args += ["--reference-csv", experiment["reference_csv"]]
    if experiment.get("plan"):
        args += ["--auto-plan", experiment["plan"]]
    for spec in experiment.get("operators", []):
        args += ["--auto-operator", spec]
    if PLAN_ONLY:
        args.append("--plan-only")
    else:
        if RUN_GENERATED:
            args.append("--run-generated")
        if REGENERATE:
            args.append("--regenerate")
        if RESET_CHECKPOINT:
            args.append("--reset-checkpoint")
        if SAVE_CLEANED:
            args.append("--save-cleaned")
    return args


def run_command(arguments):
    command = [sys.executable, str(MAIN_FILE), *arguments]
    print("\nRunning:", " ".join(map(str, command)), flush=True)
    if DRY_RUN:
        return True
    completed = subprocess.run(command, cwd=PROJECT_ROOT, check=False)
    if completed.returncode:
        print(f"Experiment failed with exit code {completed.returncode}")
        if STOP_ON_ERROR:
            raise RuntimeError("Automatic experiment stopped after an error")
        return False
    return True


def main():
    selected = check_settings()
    if not selected:
        print("No automatic experiments enabled. Set enabled=True for a dataset.")
        return

    successful = 0
    for index, experiment in enumerate(selected, start=1):
        name = experiment["name"] + (f"_{RUN_TAG}" if RUN_TAG else "")
        print(f"\n=== Auto experiment {index}/{len(selected)}: {name} ===", flush=True)

        started_ns = time.time_ns()
        if run_command(build_arguments(experiment)):
            successful += 1
            if CREATE_FIGURES and not DRY_RUN and not PLAN_ONLY:
                csv_path = PROJECT_ROOT / OUT / "auto" / name / "comparison.csv"
                if csv_path.is_file() and csv_path.stat().st_mtime_ns >= started_ns:
                    try:
                        plot_new_dataset_results(csv_path)
                    except (ValueError, OSError, ImportError) as exc:
                        print(f"[Charts] Could not plot {name}: {exc}", flush=True)
                else:
                    print(f"[Charts] No newly written comparison.csv for {name}; "
                          "skipping older results.", flush=True)

    print(f"\nCompleted: {successful}/{len(selected)} automatic experiments.")
    print(f"Results are stored separately under: {OUT}/auto/<name>/")
    print("Legacy manual experiments and their comparison files were not touched.")


if __name__ == "__main__":
    main()
