# spares_provisioning.py
from math import exp
from typing import Optional, Iterable, Dict, Any, Tuple, List
import pandas as pd

__all__ = [
    "poisson_pmf",
    "poisson_cdf",
    "poisson_ppf",
    "recommended_spares_level",
    "provision_row",
    "provision_dataframe",
    "provision_from_parameters",
    "provision_with_stations",
    "allocate_integer_proportions",
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
    p = exp(-lam)
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
# Helper(s)
# ----------------------------

def _coalesce(d: Dict[str, Any], keys: Iterable[str], default=None):
    for k in keys:
        if k in d and pd.notna(d[k]):
            return d[k]
    return default


def allocate_integer_proportions(total: int, weights: List[float]) -> List[int]:
    """
    Allocate `total` integer units across buckets according to `weights` (non-negative),
    using Largest Remainder (Hamilton) method. Returns a list of integer allocations
    summing to `total`.
    """
    if total < 0:
        raise ValueError("total must be >= 0")
    if total == 0:
        return [0] * len(weights)
    if any(w < 0 for w in weights):
        raise ValueError("weights must be non-negative")
    s = sum(weights)
    if s <= 0:
        # Equal split if all weights are zero
        weights = [1.0] * len(weights)
        s = float(len(weights))
    shares = [w / s * total for w in weights]
    base = [int(x) for x in map(float.__floor__, shares)]  # floor
    remainder = total - sum(base)
    # Assign remaining units to the largest fractional remainders
    fracs = [(i, shares[i] - base[i]) for i in range(len(weights))]
    fracs.sort(key=lambda x: x[1], reverse=True)
    for i in range(remainder):
        base[fracs[i][0]] += 1
    return base


# ----------------------------
# Provisioning core (single pool)
# ----------------------------

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


# ----------------------------
# Workbook-style parameters (adds stations & pooling mode)
# ----------------------------

def provision_from_parameters(
    aircraft_in_fleet: float,
    avg_annual_hours: float,
    qty_per_aircraft: float,
    turn_time_days: float,
    mtbf_hours: float,
    confidence_level: float,
    days_per_year: float = 360.0,
    stations: int = 1,
    mode: str = "pooled",  # "pooled" or "distributed"
    station_weights: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Mirrors the Excel logic and extends with stations:
      - qty_per_aircraft is coerced to an integer (>=1).
      - turn_time_hours = turn_time_days * (avg_annual_hours / days_per_year)
      - effective_population = aircraft_in_fleet * qty_per_aircraft
      - λ_total = effective_population * (turn_time_hours / mtbf_hours)

    Modes:
      - "pooled": compute s_pool from λ_total; optionally allocate s_pool to stations by weights.
      - "distributed": compute per-station s_i from λ_i = λ_total * weight_i; total = sum(s_i).

    Returns a dict with totals and (if stations>1) per-station breakdown.
    """
    # Validate & coerce
    qpa = max(1, int(round(qty_per_aircraft)))  # ensure whole-integer qty per aircraft
    stn = max(1, int(round(stations)))
    mode = (mode or "pooled").strip().lower()
    if mode not in ("pooled", "distributed"):
        raise ValueError("mode must be 'pooled' or 'distributed'")

    # Base calculations
    turn_time_hours = turn_time_days * (avg_annual_hours / days_per_year)
    effective_population = float(aircraft_in_fleet) * qpa
    lam_total = effective_population * (turn_time_hours / mtbf_hours)

    # Default weights
    if station_weights is None or len(station_weights) != stn:
        station_weights = [1.0] * stn
    wsum = sum(station_weights)
    if wsum <= 0:
        station_weights = [1.0] * stn
        wsum = float(stn)
    weights_norm = [w / wsum for w in station_weights]

    result: Dict[str, Any] = {
        "turn_time_hours": float(turn_time_hours),
        "effective_population": float(effective_population),
        "lambda_total": float(lam_total),
        "qty_per_aircraft_int": int(qpa),
        "stations": stn,
        "mode": mode,
        "confidence_level": float(confidence_level),
        "days_per_year": float(days_per_year),
    }

    if stn == 1:
        s_pool = poisson_ppf(confidence_level, lam_total)
        result.update({
            "recommended_spares_total": int(s_pool),
            "station_allocations": [int(s_pool)],
            "station_lambdas": [float(lam_total)],
        })
        return result

    if mode == "pooled":
        # Compute stock for the pooled system, then allocate integers across stations
        s_pool = poisson_ppf(confidence_level, lam_total)
        alloc = allocate_integer_proportions(int(s_pool), weights_norm)
        # Per-station lambdas (for reporting)
        lam_i = [lam_total * w for w in weights_norm]
        result.update({
            "recommended_spares_total": int(s_pool),
            "station_allocations": alloc,
            "station_lambdas": [float(x) for x in lam_i],
        })
    else:
        # Distributed: each station must independently meet the confidence level
        lam_i = [lam_total * w for w in weights_norm]
        s_i = [poisson_ppf(confidence_level, li) for li in lam_i]
        result.update({
            "recommended_spares_total": int(sum(s_i)),
            "station_allocations": [int(x) for x in s_i],
            "station_lambdas": [float(x) for x in lam_i],
        })

    return result


# ----------------------------
# Row-wise & DataFrame utilities
# ----------------------------

def provision_row(row: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply provisioning to a dict-like record.

    Accepted (case-insensitive) names / synonyms:
      - mtbf_hours:          'mtbf', 'mtbf_hours', 'mtbf (h)', 'mtbf [hrs]'
      - turn_time_hours:     'turn_time_hours', 'lead_time_hours', 'turn_time', 'tat', 'repair_time', 'lead_time'
      - OR (for Excel-style): 'turn_time_days' + 'avg_annual_hours' (+ 'days_per_year' optional; default 360)
      - availability_target: 'availability_target', 'availability', 'ao_target', 'service_level'
      - population:          'population', 'fleet', 'units_in_service'
      - qty_per_aircraft:    'qty_per_aircraft', 'installed_qty', 'qty_installed'  (coerced to integer >=1)
      - demand_multiplier:   'demand_multiplier', 'multiplier'
      - stations:            'stations', 'n_stations', 'stations_count'
      - mode:                'mode', 'pooling'  ('pooled' or 'distributed')
      - station_weights:     'station_weights' (comma-separated string or list)
    """
    d = {str(k).strip().lower(): v for k, v in row.items()}

    # Parse basics
    mtbf = float(_coalesce(d, ["mtbf_hours", "mtbf (h)", "mtbf [hrs]", "mtbf"]))
    avail = float(_coalesce(d, ["availability_target", "availability", "ao_target", "service_level"]))

    # qty_per_aircraft as whole integer
    qpa_val = _coalesce(d, ["qty_per_aircraft", "installed_qty", "qty_installed"], 1)
    qpa = max(1, int(round(float(qpa_val))))

    # population OR fleet; if both given, population beats fleet
    pop = _coalesce(d, ["population", "units_in_service"])
    fleet = _coalesce(d, ["fleet"])
    mult = float(_coalesce(d, ["demand_multiplier", "multiplier"], 1.0))

    # Turn time hours, or days+annual hours
    tth = _coalesce(d, ["turn_time_hours", "lead_time_hours", "turn_time", "tat", "repair_time", "lead_time"])
    if tth is None:
        ttd = float(_coalesce(d, ["turn_time_days", "tat_days"], 0.0))
        annual = float(_coalesce(d, ["avg_annual_hours", "annual_hours"], 0.0))
        dpy = float(_coalesce(d, ["days_per_year"], 360.0))
        tth = ttd * (annual / dpy)
    else:
        tth = float(tth)

    # Stations & mode
    stn = int(_coalesce(d, ["stations", "n_stations", "stations_count"], 1))
    mode = str(_coalesce(d, ["mode", "pooling"], "pooled")).strip().lower()
    # Station weights: allow list or comma-separated
    sw_raw = _coalesce(d, ["station_weights"], None)
    if isinstance(sw_raw, str):
        try:
            station_weights = [float(x) for x in sw_raw.split(",") if x.strip() != ""]
        except Exception:
            station_weights = None
    elif isinstance(sw_raw, (list, tuple)):
        station_weights = [float(x) for x in sw_raw]
    else:
        station_weights = None

    # Effective population: either directly given (population) or fleet * qty_per_aircraft
    if pop is not None:
        population = int(round(float(pop)))
    elif fleet is not None:
        population = int(round(float(fleet))) * qpa
    else:
        population = qpa  # at least 1 aircraft assumed if nothing provided

    # Apply demand multiplier to turn_time proportion (equivalent to multiplying λ)
    lam_total = population * mult * (tth / mtbf)

    # Per-station logic via the helper
    res = provision_with_stations(
        aircraft_in_fleet=float(population) / qpa if qpa else float(population),
        avg_annual_hours=0.0,          # not needed if we already have tth
        qty_per_aircraft=qpa,
        turn_time_days=0.0,            # not used when tth known
        mtbf_hours=mtbf,
        confidence_level=avail,
        days_per_year=360.0,
        stations=stn,
        mode=mode,
        station_weights=station_weights,
    )
    # Overwrite λ_total using our precomputed lam_total (includes multiplier) and recompute allocations
    res["lambda_total"] = float(lam_total)
    if stn == 1:
        s_pool = poisson_ppf(avail, lam_total)
        res["recommended_spares_total"] = int(s_pool)
        res["station_allocations"] = [int(s_pool)]
        res["station_lambdas"] = [float(lam_total)]
    else:
        if mode == "pooled":
            s_pool = poisson_ppf(avail, lam_total)
            weights = station_weights if (station_weights and len(station_weights) == stn) else [1.0] * stn
            wsum = sum(weights) or float(stn)
            weights_norm = [w / wsum for w in weights]
            alloc = allocate_integer_proportions(int(s_pool), weights_norm)
            res["recommended_spares_total"] = int(s_pool)
            res["station_allocations"] = alloc
            res["station_lambdas"] = [float(lam_total * w) for w in weights_norm]
        else:
            weights = station_weights if (station_weights and len(station_weights) == stn) else [1.0] * stn
            wsum = sum(weights) or float(stn)
            weights_norm = [w / wsum for w in weights]
            lam_i = [lam_total * w for w in weights_norm]
            s_i = [poisson_ppf(avail, li) for li in lam_i]
            res["recommended_spares_total"] = int(sum(s_i))
            res["station_allocations"] = [int(x) for x in s_i]
            res["station_lambdas"] = [float(x) for x in lam_i]

    # Also provide single-pool recommended level and lambda for convenience
    s_pool_only = poisson_ppf(avail, lam_total)
    return {
        **row,
        "qty_per_aircraft_int": int(qpa),
        "turn_time_hours": float(tth),
        "population_effective": int(population),
        "lambda_total": float(lam_total),
        "recommended_spares_total": int(res["recommended_spares_total"]),
        "station_allocations": res["station_allocations"],
        "station_lambdas": res["station_lambdas"],
        "mode": mode,
        "stations": int(stn),
        "availability_target": avail,
        "recommended_spares_pooled_only": int(s_pool_only),
    }


def provision_dataframe(
    excel_path: str,
    sheet_name: str = "ArrayFormulaBased",
    output_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Read an Excel sheet (default: 'ArrayFormulaBased'), compute recommended spares per row,
    and return a DataFrame with appended columns. Optionally writes to output_path (Excel).

    Note: If your sheet has parameters rather than row-per-SKU data, prefer `provision_from_parameters`
    or `provision_with_stations`.
    """
    df = pd.read_excel(excel_path, sheet_name=sheet_name)
    results = df.apply(lambda r: pd.Series(provision_row(r.to_dict())), axis=1)
    if output_path:
        with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
            results.to_excel(writer, index=False, sheet_name="Provisioned")
    return results


# ----------------------------
# Stations-aware API (direct)
# ----------------------------

def provision_with_stations(
    aircraft_in_fleet: float,
    avg_annual_hours: float,
    qty_per_aircraft: float,
    turn_time_days: float,
    mtbf_hours: float,
    confidence_level: float,
    days_per_year: float = 360.0,
    stations: int = 1,
    mode: str = "pooled",
    station_weights: Optional[List[float]] = None,
) -> Dict[str, Any]:
    """
    Same as provision_from_parameters, but returns a full breakdown regardless of `stations`.
    Use this if you want explicit control and a structured result every time.
    """
    return provision_from_parameters(
        aircraft_in_fleet=aircraft_in_fleet,
        avg_annual_hours=avg_annual_hours,
        qty_per_aircraft=qty_per_aircraft,
        turn_time_days=turn_time_days,
        mtbf_hours=mtbf_hours,
        confidence_level=confidence_level,
        days_per_year=days_per_year,
        stations=stations,
        mode=mode,
        station_weights=station_weights,
    )


# ----------------------------
# Optional CLI for quick tests
# ----------------------------
if __name__ == "__main__":
    import argparse, json
    p = argparse.ArgumentParser(description="Poisson-based aircraft spares provisioning with stations")
    p.add_argument("--fleet", type=float, required=True, help="Aircraft in fleet")
    p.add_argument("--annual_hours", type=float, required=True, help="Average annual hours per aircraft")
    p.add_argument("--qty_per_aircraft", type=float, required=True, help="Qty required per aircraft (will be rounded to integer)")
    p.add_argument("--tat_days", type=float, required=True, help="Turn Around Time (days)")
    p.add_argument("--mtbf_hours", type=float, required=True, help="MTBF (hours)")
    p.add_argument("--confidence", type=float, required=True, help="Confidence (service level), e.g. 0.98")
    p.add_argument("--days_per_year", type=float, default=360.0, help="Utilization divisor (360 or 365)")
    p.add_argument("--stations", type=int, default=1, help="Number of stations")
    p.add_argument("--mode", type=str, default="pooled", choices=["pooled", "distributed"], help="Pooling mode")
    p.add_argument("--station_weights", type=str, default="", help="Comma-separated weights for stations (optional)")

    args = p.parse_args()
    weights = None
    if args.station_weights.strip():
        weights = [float(x) for x in args.station_weights.split(",") if x.strip() != ""]

    res = provision_from_parameters(
        aircraft_in_fleet=args.fleet,
        avg_annual_hours=args.annual_hours,
        qty_per_aircraft=args.qty_per_aircraft,
        turn_time_days=args.tat_days,
        mtbf_hours=args.mtbf_hours,
        confidence_level=args.confidence,
        days_per_year=args.days_per_year,
        stations=args.stations,
        mode=args.mode,
        station_weights=weights,
    )
    print(json.dumps(res, indent=2))
