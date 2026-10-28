import re

def clean_row(row):
    if not isinstance(row, dict):
        return {"city": None, "state": None}

    city = row.get("city")
    state = row.get("state")

    city = city.strip() if isinstance(city, str) else None
    state = state.strip() if isinstance(state, str) else None
    city = city or None
    state = state or None

    state_codes = {
        "AL", "AK", "AZ", "AR", "CA", "CO", "CT", "DE", "FL", "GA",
        "HI", "ID", "IL", "IN", "IA", "KS", "KY", "LA", "ME", "MD",
        "MA", "MI", "MN", "MS", "MO", "MT", "NE", "NV", "NH", "NJ",
        "NM", "NY", "NC", "ND", "OH", "OK", "OR", "PA", "RI", "SC",
        "SD", "TN", "TX", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
        "DC", "AS", "GU", "MP", "PR", "VI"
    }

    if state and state.upper() in state_codes:
        state = state.upper()

    if city:
        match = re.fullmatch(r"(.+?)(?:\s*,\s*|\s+)([A-Z]{2})", city)
        if match:
            city_part = match.group(1).strip()
            code = match.group(2)
            if city_part and code in state_codes:
                if state is None or state == code:
                    city = city_part
                    state = code

    return {"city": city, "state": state}