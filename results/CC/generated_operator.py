def clean_row(row):
    if not isinstance(row, dict):
        return {"city": None}

    value = row.get("city")
    if not isinstance(value, str):
        return {"city": None}

    city = " ".join(value.split()).lower()
    if not city:
        return {"city": None}

    canonical_cities = (
        "boaz", "ozark", "huntsville", "gadsden", "andalusia",
        "florence", "anniston", "georgiana", "fayette",
        "centre", "dothan"
    )

    if city in canonical_cities:
        return {"city": city}

    if "x" in city:
        matches = [
            candidate
            for candidate in canonical_cities
            if len(candidate) == len(city)
            and all(
                actual == expected or actual == "x"
                for actual, expected in zip(city, candidate)
            )
        ]
        if len(matches) == 1:
            return {"city": matches[0]}

    return {"city": city}