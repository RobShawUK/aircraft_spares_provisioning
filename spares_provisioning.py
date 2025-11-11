from math import exp
from typing import Optional, Iterable, Dict, Any, Tuple, List
import pandas as pd

# ---------------------------------------------------------
# Public API exported when using: from spares_provisioning import *
# ---------------------------------------------------------
__all__ = [
    "poisson_pmf",
    "poisson_cdf",
    "poisson_ppf",
    "recommended_spares_level",
    "provision_row",
    "provision_dataframe",
    "provision_from_parameters",
    "network_lambda",
    "allocate_integers_lrm",
    "provision_network",
]

# ---------------------------------------------------------
# Basic Poisson distribution utilities
# ---------------------------------------------------------

def poisson_pmf(k: int, lam: float) -> float:
    """Poisson PMF: P(K = k) for rate lam."""
    if k < 0:
        return 0.0
    if lam < 0:
        raise ValueError("lam must be >= 0")
    if k == 0:
        return exp(-lam)
    p = exp(-lam)
    for i in range(1, k + 1):
        p *= lam / i
    return p


def poisson_cdf(k: int, lam: float) -> float:
    """Poisson CDF: P(K ≤ k)."""
    if k < 0:
        return 0.0
    total = 0.0
    term = exp(-lam)
    total += term
    for i in range(1, k + 1):
        term *= lam / i
        total += term
    return min(1.0, total)


def poisson_ppf(p: float, lam: float) -> int:
    """
    Percent-point function (inverse CDF).
    Returns smallest integer k such that P(K ≤ k) ≥ p.
    """
    if not (0.0 < p <= 1.0):
        raise ValueError("p must be in (0,1]")
    if lam < 0:
        raise ValueError("lam must be >= 0")
    k = 0
    while poisson_cdf(k, lam) < p:
        k += 1
    return k

# ---------------------------------------------------------
# Core provisioning logic
# ---------------------------------------------------------

def _coalesce(d: Dict[str, Any], keys: Iterable[str], default=None):
    for k in keys:
        if k in d and pd.notna(d[k]):
            return d[k]
    return default


def recommended_spares_level(
    mtbf_hours: float,
    turn_time_hours: float,
    availability_target: float,
    population: int = 1,
    demand_multiplier: float = 1.0,
) -> Tuple[int, float]:
    """
    Compute s such that:
        P{ Poisson(λ) ≤ s } ≥ availability_target
    where λ = population × (turn_time_hours / mtbf_hours)
    """
    if mtbf_hours <= 0 or turn_time_hours < 0 or population <= 0:
        raise ValueError("Invalid parameters")
    if not (0.5 <= availability_target <= 0.999999):
        raise ValueError("availability_target should be in [0.5, 0.999999]")

    lam = population * demand_multiplier * (turn_time_hours / mtbf_hours)
    s = poisson_ppf(availability_target, lam)
    return s, lam


def provision_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Row-wise provisioning from a dataframe.
    Accepts flexible column names (case-insensitive).
    """
    d = {str(k).strip().lower(): v for k, v in row.items()}
    mtbf = float(_coalesce(d, ["mtbf_hours", "mtbf (h)", "mtbf [hrs]", "mtbf"]))
    turn = float(_coalesce(d, [
        "turn_time_hours", "lead_time_hours",
        "turn_time", "tat", "repair_time", "lead_time"
    ]))
    avail = float(_coalesce(d, [
        "availability_target", "availability", "ao_target", "service_level"
    ]))
    pop = int(_coalesce(d, ["population", "fleet", "units_in_service"], 1))
    mult = float(_coalesce(d, ["demand_multiplier", "multiplier"], 1.0))

    s, lam = recommended_spares_level(mtbf, turn, avail, pop, mult)
    return {
        **row,
        "lambda_demand": lam,
        "expected_demand_during_turn": lam,
        "recommended_spares": int(s),
        "availability_target": avail,
    }


def provision_dataframe(
    excel_path: str,
    sheet_name: str = "ArrayFormulaBased",
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    results = df.apply(lambda r: pd.Series(provision_row(r.to_dict())), axis=1)
    if output_path:
        with pd.ExcelWriter(output_path, engine="openpyxl") as w:
            results.to_excel(w, index=False, sheet_name="Provisioned")
    return results

# ---------------------------------------------------------
# Excel-like parameter model
# ---------------------------------------------------------

def provision_from_parameters(
    aircraft_in_fleet: float,
    avg_annual_hours: float,
    qty_per_aircraft: float,
    turn_time_days: float,
    mtbf_hours: float,
    confidence_level: float,
    days_per_year: float = 360.0,
) -> Dict[str, float]:
    """Reproduces your Excel model exactly using 360-day year logic."""
    turn_time_hours = turn_time_days * (avg_annual_hours / days_per_year)
    effective_population = aircraft_in_fleet * qty_per_aircraft
    lam = effective_population * (turn_time_hours / mtbf_hours)
    s = poisson_ppf(confidence_level, lam)
    return {
        "turn_time_hours": float(turn_time_hours),
        "effective_population": float(effective_population),
        "lambda": float(lam),
        "recommended_spares": int(s),
    }

# ---------------------------------------------------------
# Network logic: pooled vs distributed
# ---------------------------------------------------------

def network_lambda(
    aircraft_in_fleet: float,
    qty_per_aircraft: float,
    turn_time_hours: float,
    mtbf_hours: float,
) -> float:
    """Total λ across entire network."""
    effective_population = aircraft_in_fleet * qty_per_aircraft
    return effective_population * (turn_time_hours / mtbf_hours)


def allocate_integers_lrm(targets: List[float], total: int) -> List[int]:
    """
    Largest Remainder Method for integer allocation.
    targets = weights; total = total integer budget.
    """
    if total < 0:
        raise ValueError("total must be >= 0")
    if any(t < 0 for t in targets):
        raise ValueError("targets must be non-negative")

    s = sum(targets)
    if s == 0:
        base = [0] * len(targets)
        for i in range(total):
            base[i % len(base)] += 1
        return base

    raw = [t / s * total for t in targets]
    base = [int(x) for x in raw]
    rem = total - sum(base)

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
    mode: str = "pooled",                # "pooled" or "distributed"
    station_weights: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Network-aware provisioning:
    - POOLED: one shared pool (best risk pooling)
    - DISTRIBUTED: per-station stock to meet SAME service level locally
    """
    if stations <= 0:
        raise ValueError("stations must be >= 1")

    turn_time_hours = turn_time_days * (avg_annual_hours / days_per_year)
    lam_total = network_lambda(
        aircraft_in_fleet, qty_per_aircraft, turn_time_hours, mtbf_hours
    )

    # ------------- Pooled -------------
    S_pool = poisson_ppf(confidence_level, lam_total)

    # ------------- Distributed -------------
    if station_weights is None:
        lam_i = [lam_total / stations] * stations
    else:
        if len(station_weights) != stations:
            raise ValueError("station_weights length mismatch")
        if any(w < 0 for w in station_weights):
            raise ValueError("station_weights must be non-negative")
        ws = sum(station_weights)
        lam_i = [(lam_total / ws) * w for w in station_weights]

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
        "pooling_savings": int(S_dist - S_pool),   # >0 means distributed needs more stock
    }

# ---------------------------------------------------------
# Optional CLI
# ---------------------------------------------------------
if __name__ == "__main__":
    print("This module is intended for import, not direct execution.")
