def pt_num(value: float, decimals: int = 1) -> str:
    return f"{value:.{decimals}f}".replace(".", ",")


def pt_pct(fraction: float, decimals: int = 1) -> str:
    return pt_num(fraction * 100, decimals) + "%"


def pt_signed(value: float, decimals: int = 1) -> str:
    return ("+" if value >= 0 else "") + pt_num(value, decimals)


def pt_int(value: int) -> str:
    return f"{value:,}".replace(",", ".")
