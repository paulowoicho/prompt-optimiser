def exact_match(expected: str, predicted: str) -> float:
    """Whitespace-trimmed, case-sensitive exact match in [0, 1]."""
    return float(expected.strip() == predicted.strip())
