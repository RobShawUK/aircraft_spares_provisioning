from math import exp, sqrt
from typing import Optional, Iterable, Dict, Any, Tuple
import pandas as pd

__all__ = [
    "poisson_pmf",
    "poisson_cdf",
    "poisson_ppf",
    "recommended_spares_level",
    "provision_row",
    "provision_dataframe",
    "provision_from_parameters",
]

# ----------------------------
# Poisson utilities (no SciPy)
# ----------------------------

def poisson_pmf(k: int, lam: float) -> float:
    """Poisson probability mass function P(K = k) with rate λ."""
    if k < 0:
        return 0.0
    if lam < 0:
        raise ValueError("lam must be >= 0")
    if k == 0:
        return exp(-lam)
    p0 = exp(-lam)
    p = p0
    for i in range(1, k + 1):
        p *= lam / i
    return p


def poisson_cdf(k: int, lam: float) -> float:
    """Poisson cumulative distribution function P(K <= k)."""
    if k < 0:
        return 0.0
    total = 0.0
    term = exp(-lam)  # P(0)
    total += term
    for i in range(1, k + 1):
        term *= lam / i
        total += term
    return min(1.0, total)


def poisson_ppf(p: float, lam: float) -> int:
    """
    Percent-point function (inverse CDF / quantile).
    Returns the smallest integer k such that P(K <= k) >= p.
    """
    if not (0.0 < p <= 1.0):
        raise ValueError("p must be in (0, 1]")
    if lam < 0:
        raise ValueError("lam must be >= 0")

    # Simple increasing search is fine for typical provisioning λ (usually small).
    k = 0
    while poisson_cdf(k, lam) < p:
        k += 1
    return k


# ----------------------------
# Provisioning core
# ----------------------------

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
    Compute base-stock `s` for a single SKU so that:
        P{ demand during turn-time ≤ s } ≥ availability_target,
    where demand ~ Poisson(λ) and λ = population * demand_multiplier * (turn_time_hours / mtbf_hours).

    Returns:
        (s, lam) where s is the integer stock level and lam is the Poisson rate used.
    """
    if mtbf_hours <= 0 or turn_time_hours < 0 or population <= 0:
        raise ValueError("mtbf_hours>0, turn_time_hours>=0, population>0 required")
    if not (0.5 <= availability_target <= 0.999999):
        # Typical service levels are 0.8–0.99+; allow a wide but sensible range
        raise ValueError("availability_target should be in [0.5, 0.999999]")

    lam = population * demand_multiplier * (turn_time_hours / mtbf_hours)
    s = poisson_ppf(availability_target, lam)
    return s, lam


def provision_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply recommended_spares_level to a dict-like record.
    Accepted (case-insensitive) column names / synonyms:

      - mtbf_hours:          'mtbf', 'mtbf_hours', 'mtbf (h)', 'mtbf [hrs]'
      - turn_time_hours:     'turn_time_hours', 'lead_time_hours', 'turn_time', 'tat', 'repair_time', 'lead_time'
      - availability_target: 'availability_target', 'availability', 'ao_target', 'service_level'
      - population:          'population', 'fleet', 'units_in_service'          (default: 1)
      - demand_multiplier:   'demand_multiplier', 'multiplier'                  (default: 1.0)
    """
    d = {str(k).strip().lower(): v for k, v in row.items()}

    mtbf = float(_coalesce(d, ["mtbf_hours", "mtbf (h)", "mtbf [hrs]", "mtbf"]))
    turn = float(_coalesce(d, ["turn_time_hours", "lead_time_hours", "turn_time", "tat", "repair_time", "lead_time"]))
    avail = float(_coalesce(d, ["availability_target", "availability", "ao_target", "service_level"]))
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
    """
    Read an Excel sheet (default: 'ArrayFormulaBased'), compute recommended spares per row,
    and return a DataFrame with appended columns. Optionally writes to output_path (Excel).

    Note: If your sheet has parameters rather than row-per-SKU data, prefer `provision_from_parameters`.
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    results = df.apply(lambda r: pd.Series(provision_row(r.to_dict())), axis=1)
    if output_path:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            results.to_excel(writer, index=False, sheet_name="Provisioned")
    return results


def provision_from_parameters(
    aircraft_in_fleet: float,
    avg_annual_hours: float,
    qty_per_aircraft: float,
    turn_time_days: float,
    mtbf_hours: float,
    confidence_level: float,
    days_per_year: float = 360.0,
) -> Dict[str, float]:
    """
    Mirrors the Excel 'ArrayFormulaBased' logic using a 360-day conversion by default:

        turn_time_hours = turn_time_days * (avg_annual_hours / days_per_year)
        effective_population = aircraft_in_fleet * qty_per_aircraft
        λ = effective_population * (turn_time_hours / mtbf_hours)
        s = min { k ∈ ℕ : P{Poisson(λ) ≤ k} ≥ confidence_level }

    Returns a dict with summary fields.
    """
    required = [aircraft_in_fleet, avg_annual_hours, qty_per_aircraft, turn_time_days, mtbf_hours, confidence_level]
    if any(x is None for x in required):
        raise ValueError("All parameters must be provided")

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


# ----------------------------
# Optional CLI for quick tests
# ----------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Poisson-based spares provisioning")
    parser.add_argument("--fleet", type=float, required=True, help="Aircraft in fleet")
    parser.add_argument("--annual_hours", type=float, required=True, help="Average annual hours per aircraft")
    parser.add_argument("--qty_per_aircraft", type=float, required=True, help="Qty required per aircraft")
    parser.add_argument("--tat_days", type=float, required=True, help="Turn Around Time (days)")
    parser.add_argument("--mtbf_hours", type=float, required=True, help="MTBF (hours)")
    parser.add_argument("--confidence", type=float, required=True, help="Confidence (service level), e.g. 0.98")
    parser.add_argument("--days_per_year", type=float, default=360.0, help="Utilization divisor (360 or 365)")

    args = parser.parse_args()
    res = provision_from_parameters(
        aircraft_in_fleet=args.fleet,
        avg_annual_hours=args.annual_hours,
        qty_per_aircraft=args.qty_per_aircraft,
        turn_time_days=args.tat_days,
        mtbf_hours=args.mtbf_hours,
        confidence_level=args.confidence,
        days_per_year=args.days_per_year,
    )
    print(res)
