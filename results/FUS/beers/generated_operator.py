import re
import math

def clean_row(row):
    result = dict(row) if isinstance(row, dict) else {}
    value = result.get("ounces")
    cleaned = None

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        try:
            number = float(value)
            if math.isfinite(number):
                cleaned = number
        except (TypeError, ValueError, OverflowError):
            pass
    elif isinstance(value, str):
        match = re.fullmatch(
            r"\s*([+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?)\s*(?:(?:oz|ounces?)\.?)?\s*",
            value,
            re.IGNORECASE,
        )
        if match:
            try:
                number = float(match.group(1))
                if math.isfinite(number):
                    cleaned = number
            except (TypeError, ValueError, OverflowError):
                pass

    result["ounces"] = cleaned
    return result