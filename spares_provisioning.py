# --- add below your existing imports & __all__ ---
from typing import List

__all__ += [
    "network_lambda",
    "provision_network",
    "allocate_integers_lrm",
]

def network_lambda(
    aircraft_in_fleet: float,
    qty_per_aircraft: float,
    turn_time_hours: float,
    mtbf_hours: float,
) -> float:
    """
    Total lambda (demand during turn) across the whole network if inventory is pooled.
    """
    effective_population = aircraft_in_fleet * qty_per_aircraft
    return effective_population * (turn_time_hours / mtbf_hours)


def allocate_integers_lrm(targets: List[float], total: int) -> List[int]:
    """
    Largest Remainder Method: integer allocation that sums to `total`,
    proportional to `targets` (non-negative).
    """
    if total < 0:
        raise ValueError("total must be >= 0")
    if any(t < 0 for t in targets):
        raise ValueError("targets must be non-negative")
    s = sum(targets)
    if s == 0:
        base = [0] * len(targets)
        # spread evenly
        for i in range(total):
            base[i % len(base)] += 1
        return base

    raw = [t / s * total for t in targets]
    base = [int(x) for x in raw]
    rem = total - sum(base)
    # Distribute remainders to largest fractional parts
    frac_idx = sorted(range(len(raw)), key=lambda i: raw[i] - base[i], reverse=True)
    for i in range(rem):
        base[frac_idx[i % len(frac_idx)]] += 1
    return base


def provision_network(
    aircraft_in_fleet: float,
    avg_annual_hours: float,
    qty_per_aircraft: float,
    turn_time_days: float,
    mtbf_hours: float,
    confidence_level: float,
    stations: int = 1,
    days_per_year: float = 360.0,
    mode: str = "pooled",                  # "pooled" or "distributed"
    station_weights: Optional[List[float]] = None,  # relative demand weights per station
    allocation: str = "proportional",      # "proportional" or "equal" for distributed split of TOTAL S
) -> Dict[str, Any]:
    """
    Network-aware provisioning.

    POOLED:
        Compute a single stock S_pool for the whole network using total λ.
        (Best service for a given total stock, thanks to risk pooling.)

    DISTRIBUTED:
        Two common choices:
        (A) Per-station service: compute s_i for each station using λ_i and the same confidence_level.
            If station_weights are omitted, split λ evenly (identical stations).
            Total stock S_dist = sum_i s_i. (Typically > S_pool.)
        (B) If you prefer to fix S_total and then split it: set allocation="proportional" or "equal"
            and pass station_weights. (We keep it simple and default to (A).)

    Returns:
        {
          "turn_time_hours", "lambda_total",
          "pooled": {"recommended_spares": S_pool, "lambda": lambda_total},
          "distributed": {
              "lambda_per_station": [..],
              "recommended_spares_per_station": [..],
              "recommended_spares_total": S_dist
          }
        }
    """
    if stations <= 0:
        raise ValueError("stations must be >= 1")

    # Convert to hours & compute network λ
    turn_time_hours = turn_time_days * (avg_annual_hours / days_per_year)
    lam_total = network_lambda(
        aircraft_in_fleet, qty_per_aircraft, turn_time_hours, mtbf_hours
    )

    # ---- Pooled (single pool) ----
    S_pool = poisson_ppf(confidence_level, lam_total)

    # ---- Distributed (per-station) ----
    if station_weights is None:
        # Identical stations
        lam_i = [lam_total / stations] * stations
    else:
        if len(station_weights) != stations:
            raise ValueError("station_weights length must equal stations")
        if any(w < 0 for w in station_weights):
            raise ValueError("station_weights must be non-negative")
        wsum = sum(station_weights)
        if wsum == 0:
            lam_i = [lam_total / stations] * stations
        else:
            lam_i = [lam_total * (w / wsum) for w in station_weights]

    s_i = [poisson_ppf(confidence_level, lam) for lam in lam_i]
    S_dist = int(sum(s_i))

    return {
        "turn_time_hours": float(turn_time_hours),
        "lambda_total": float(lam_total),
        "pooled": {
            "recommended_spares": int(S_pool),
            "lambda": float(lam_total),
        },
        "distributed": {
            "lambda_per_station": [float(x) for x in lam_i],
            "recommended_spares_per_station": [int(x) for x in s_i],
            "recommended_spares_total": int(S_dist),
        },
        "pooling_savings": int(S_dist - S_pool),  # positive number = extra stock needed when distributed
    }
