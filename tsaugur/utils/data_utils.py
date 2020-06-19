def period_to_int(period):
    """
    Reference: https://robjhyndman.com/hyndsight/seasonal-periods/
    :param period:
    :return:
    """
    mapper = {
        "annual": 1,
        "a": 1,
        "quarterly": 4,
        "q": 4,
        "daily": 7,
        "d": 7,
        "monthly": 12,
        "m": 12,
        "hourly": 24,
        "h": 24,
        "weekly": 52,
        "w": 52,
    }
    if period not in mapper.keys():
        raise ValueError(f"{period} is not a valid value for the 'period' argument.")
    return mapper[period]
