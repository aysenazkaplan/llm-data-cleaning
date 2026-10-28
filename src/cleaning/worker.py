

import importlib.util
import json
import sys


def main():
    spec = importlib.util.spec_from_file_location(
        "generated_operator",
        sys.argv[1]
    )

    if spec is None or spec.loader is None:
        raise ValueError("Could not load operator")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not callable(
            getattr(module, "clean_row", None)
    ):
        raise ValueError("clean_row missing")

    records = json.loads(sys.stdin.read())
    outputs = []

    for record in records:
        try:
            result = module.clean_row(
                record["input"]
            )

            outputs.append({
                "index": record["index"],
                "output": result,
                "error": None
            })

        except Exception as exc:
            outputs.append({
                "index": record["index"],
                "output": None,
                "error": (
                    f"{type(exc).__name__}: {exc}"
                )
            })

    print(json.dumps(outputs, allow_nan=False))


if __name__ == "__main__":
    main()
