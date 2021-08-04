def _assert_same_length(x, y, x_str, y_str):
    if len(x) != len(y):
        raise ValueError(
            f"size of {x_str} ({len(x)}) is different from {y_str} ({len(y)})"
        )
