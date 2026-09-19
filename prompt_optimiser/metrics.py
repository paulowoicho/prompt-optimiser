"""Numeric metrics for prompt selection and evaluation."""


def exact_match(expected: str, predicted: str) -> float:
    """Compare texts after stripping surrounding whitespace.

    Args:
        expected: Reference text.
        predicted: Model output.

    Returns:
        1.0 for a case-sensitive match, otherwise 0.0.
    """
    return float(expected.strip() == predicted.strip())
