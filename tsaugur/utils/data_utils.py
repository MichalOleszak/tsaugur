def _period_to_int(period):
    """
    Convert time series' period from string representation to integer.
    :param period: Int or Str, the number of observations per cycle: 1 or "annual" for yearly data, 4 or "quarterly"
    for quarterly data, 7 or "daily" for daily data, 12 or "monthly" for monthly data, 24 or "hourly" for hourly
    data, 52 or "weekly" for weekly data. First-letter abbreviations of strings work as well ("a", "q", "d", "m",
    "h" and "w", respectively). Additional reference: https://robjhyndman.com/hyndsight/seasonal-periods/.
    :return: Int, a time series' period.
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
