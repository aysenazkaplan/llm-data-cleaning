
import subprocess
import sys

from pathlib import Path


EXPERIMENTS = [
    {
        "task": "FUS",
        "dataset": "beers",
        "enabled": True
    },
    {
        "task": "FUS",
        "dataset": "hospital",
        "enabled": True
    },
    {
        "task": "CFC",
        "dataset": "beers",
        "enabled": True
    },
    {
        "task": "NFN",
        "dataset": "beers",
        "enabled": True
    },
    {
        "task": "CC",
        "dataset": "hospital",
        "enabled": True
    }
]

MODE = "both"
# "direct", "synthesize", "both"
EXAMPLES = 15
EVAL_ROWS = "all"
BATCH_SIZE = 20
SEED = 42
AFFECTED_ONLY = False
RUN_GENERATED = True
REGENERATE = False
RESET_CHECKPOINT = False
COMPARE_ALL_AFTER = True
STOP_ON_ERROR = True


PROJECT_ROOT = (
    Path(__file__).resolve().parent
)

MAIN_FILE = (
        PROJECT_ROOT / "main.py"
)


def run_command(arguments):

    command = [
        sys.executable,
        str(MAIN_FILE),
        *arguments
    ]

    print(
        "\nRunning:",
        " ".join(command),
        flush=True
    )

    completed = subprocess.run(
        command,
        cwd=PROJECT_ROOT,
        check=False
    )

    if completed.returncode != 0:
        print(
            "Failed with exit code "
            f"{completed.returncode}. "
            "Restart with the same settings "
            "to resume."
        )

        if STOP_ON_ERROR:
            raise RuntimeError(
                "Experiment stopped. "
                "Saved checkpoint is retained."
            )

        return False

    return True


def run_experiment(experiment):

    arguments = [
        "--task",
        experiment["task"],

        "--dataset",
        experiment["dataset"],

        "--mode",
        MODE,

        "--examples",
        str(EXAMPLES),

        "--eval-rows",
        str(EVAL_ROWS),

        "--batch-size",
        str(BATCH_SIZE),

        "--seed",
        str(SEED)
    ]

    if AFFECTED_ONLY:
        arguments.append(
            "--affected-only"
        )

    if RUN_GENERATED:
        arguments.append(
            "--run-generated"
        )

    if REGENERATE:
        arguments.append(
            "--regenerate"
        )

    if RESET_CHECKPOINT:
        arguments.append(
            "--reset-checkpoint"
        )

    return run_command(
        arguments
    )


def main():

    selected = [
        experiment
        for experiment in EXPERIMENTS
        if experiment["enabled"]
    ]

    if not selected:
        print(
            "No experiments enabled. "
            "Use compare_results.py for "
            "comparison only."
        )

        return

    for index, experiment in enumerate(
            selected,
            start=1
    ):
        print(
            f"\n=== Experiment "
            f"{index}/{len(selected)}: "
            f"{experiment['task']} / "
            f"{experiment['dataset']} ===",
            flush=True
        )

        success = run_experiment(
            experiment
        )

        if not success:
            print(
                "Skipping automatic comparison "
                "because an experiment failed."
            )

            return

    if COMPARE_ALL_AFTER:
        run_command(
            ["--compare-all"]
        )

    print(
        "\nExperiment workflow completed."
    )


if __name__ == "__main__":
    main()
