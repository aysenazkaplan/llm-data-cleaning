import re
import math

def clean_row(row):
    value = row.get("abv") if isinstance(row, dict) else None
    if value is None or isinstance(value, bool):
        return {"abv": None}
    if isinstance(value, str):
        value = value.strip()
        if value.endswith("%"):
            value = value[:-1].strip()
        if not re.fullmatch(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?", value):
            return {"abv": None}
    elif not isinstance(value, (int, float)):
        return {"abv": None}
    try:
        value = float(value)
        return {"abv": value if math.isfinite(value) else None}
    except (ValueError, TypeError, OverflowError):
        return {"abv": None}