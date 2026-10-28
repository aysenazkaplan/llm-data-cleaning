import re
import math


def clean_row(row):
    result = dict(row) if isinstance(row, dict) else {}
    value = result.get("Score")

    if value is None:
        result["Score"] = None
        return result

    if isinstance(value, float) and math.isnan(value):
        result["Score"] = None
        return result

    if not isinstance(value, (str, int, float)) or isinstance(value, bool):
        result["Score"] = str(value)
        return result

    text = str(value).strip()
    if not text:
        result["Score"] = None
        return result

    if re.fullmatch(r"x[0-9]\s*%", text):
        text = "9" + text[1] + "%"
    elif re.fullmatch(r"(?:1xx|x00)\s*%", text):
        text = "100%"
    elif re.fullmatch(r"9x\s*%", text):
        text = "97%"
    elif re.fullmatch(r"[0-9]+(?:\.[0-9]+)?\s*x", text):
        text = text[:-1].strip() + "%"
    elif re.fullmatch(r"[0-9]+(?:\.[0-9]+)?\s*%", text):
        text = text[:-1].strip() + "%"

    result["Score"] = text
    return result