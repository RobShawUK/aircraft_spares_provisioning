# app.py — Spares Provisioning (Poisson) with Pooled vs Distributed stations

import math
import io
import pandas as pd
import streamlit as st

# Import from your module
from spares_provisioning import (
    recommended_spares_level,  # fallback path
    # provision_from_parameters may or may not exist depending on your module version
)

st.set_page_config(page_title="Spares Provisioning (Poisson)", page_icon="✈️", layout="centered")
st.title("✈️ Spares Provisioning (Poisson)")

with st.expander("How it works", expanded=False):
    st.markdown(
        """
This tool sizes spares using a Poisson model for demand during the **Turn-Around Time (TAT)** window.

**Core math**
- Effective population = `fleet × qty_per_aircraft`
- TAT hours = `TAT days × (annual hours / days_per_year)` (360 or 365)
- λ (demand during turn) = `effective_population × (TAT hours / MTBF)`
- We choose the smallest integer **s** such that **P{Poisson(λ) ≤ s} ≥ Confidence Level**

**Pooling strategies**
- **Pooled**: one central pool; compute spares on the total λ.
- **Distributed**: split the fleet across N stations; compute each station’s spares independently and sum them up (no trans-ship pooling benefit).
        """
    )

# ---------- Helpers ----------

def try_provision_from_parameters(
    fleet: int,
    qty_per_ac: int,
    annual_hours: float,
    tat_days: float,
    mtbf_hours: float,
    conf: float,
    days_per_year: float,
):
    """
    Robust wrapper:
    1) Try to use spares_provisioning.provision_from_parameters if present.
    2) Fallback to computing via recommended_spares_level with computed turn_time_hours.
    Returns a dict with keys: turn_time_hours, effective_population, lambda, recommended_spares
    """
    # Compute derived values
    turn_time_hours = tat_days * (annual_hours / days_per_year)
    population = int(fleet * qty_per_ac)

    # Try to call provision_from_parameters if available
    try:
        from spares_provisioning import provision_from_parameters  # type: ignore
        res = provision_from_parameters(
            aircraft_in_fleet=fleet,
            avg_annual_hours=annual_hours,
            qty_per_aircraft=qty_per_ac,
            turn_time_days=tat_days,
            mtbf_hours=mtbf_hours,
            confidence_level=conf,
            days_per_year=days_per_year,
        )
        # Ensure consistent keys & types
        return {
            "turn_time_hours": float(res.get("turn_time_hours", turn_time_hours)),
            "effective_population": float(res.get("effective_population", population)),
            "lambda": float(res.get("lambda")),
            "recommended_spares": int(res.get("recommended_spares")),
        }
    except Exception:
        # Fallback using recommended_spares_level
        s, lam = recommended_spares_level(
            mtbf_hours=mtbf_hours,
            turn_time_hours=turn_time_hours,
            availability_target=conf,
            population=population,
            demand_multiplier=1.0,
        )
        return {
            "turn_time_hours": float(turn_time_hours),
            "effective_population": float(population),
            "lambda": float(lam),
            "recommended_spares": int(s),
        }

def split_fleet_across_stations(fleet: int, n_stations: int):
    """
    Return a list of per-station fleet counts that sum to fleet and are as balanced as possible.
    Example: fleet=10, stations=3 -> [4, 3, 3]
    """
    base = fleet // n_stations
    rem = fleet % n_stations
    return [base + (1 if i < rem else 0) for i in range(n_stations)]

def summarize_pooled(
    fleet: int, qty_per_ac: int, mtbf: float, annual_hours: float,
    tat_days: float, conf: float, days_per_year: float
):
    res = try_provision_from_parameters(
        fleet=fleet,
        qty_per_ac=qty_per_ac,
        annual_hours=annual_hours,
        tat_days=tat_days,
        mtbf_hours=mtbf,
        conf=conf,
        days_per_year=days_per_year,
    )
    s_total = int(res["recommended_spares"])
    summary = {
        "Strategy": "Pooled",
        "Fleet": fleet,
        "Qty per aircraft": qty_per_ac,
        "MTBF (hours)": mtbf,
        "Annual hours": annual_hours,
        "TAT (days)": tat_days,
        "Days/year": days_per_year,
        "Confidence": conf,
        "Turn time (hours)": round(res["turn_time_hours"], 6),
        "Effective population (fleet × qty/AC)": int(res["effective_population"]),
        "Lambda": round(res["lambda"], 6),
        "Recommended spares (total)": s_total,
    }
    df = pd.DataFrame([summary])
    return summary, s_total, df

def summarize_distributed(
    fleet: int, qty_per_ac: int, mtbf: float, annual_hours: float,
    tat_days: float, conf: float, days_per_year: float, n_stations: int
):
    parts_by_station = split_fleet_across_stations(fleet, n_stations)
    rows = []
    total_spares = 0
    for i, fleet_i in enumerate(parts_by_station, start=1):
        res_i = try_provision_from_parameters(
            fleet=fleet_i,
            qty_per_ac=qty_per_ac,
            annual_hours=annual_hours,
            tat_days=tat_days,
            mtbf_hours=mtbf,
            conf=conf,
            days_per_year=days_per_year,
        )
        s_i = int(res_i["recommended_spares"])
        total_spares += s_i
        rows.append({
            "Station": i,
            "Fleet at station": int(fleet_i),
            "Qty per aircraft": qty_per_ac,
            "Turn time (hours)": round(res_i["turn_time_hours"], 6),
            "Effective population": int(res_i["effective_population"]),
            "Lambda": round(res_i["lambda"], 6),
            "Recommended spares (station)": s_i,
        })
    df = pd.DataFrame(rows)
    summary = {
        "Strategy": f"Distributed across {n_stations} stations",
        "Fleet": fleet,
        "Qty per aircraft": qty_per_ac,
        "MTBF (hours)": mtbf,
        "Annual hours": annual_hours,
        "TAT (days)": tat_days,
        "Days/year": days_per_year,
        "Confidence": conf,
        "Total spares (sum over stations)": int(total_spares),
    }
    return summary, int(total_spares), df

# ---------- UI ----------

tab1, tab2 = st.tabs(["Single scenario", "Batch (CSV/XLSX)"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        fleet = st.number_input("Aircraft in fleet", min_value=1, step=1, value=50)
        qty_per_ac = st.number_input("Qty required per aircraft", min_value=1, step=1, value=2)
        mtbf = st.number_input("MTBF (hours)", min_value=1.0, value=40000.0, step=100.0)
    with c2:
        annual_hours = st.number_input("Average annual hours per aircraft", min_value=1.0, value=2000.0, step=50.0)
        tat_days = st.number_input("Turn Around Time (days)", min_value=0.0, value=25.0, step=1.0)
        conf = st.slider("Confidence Level (service level)", 0.50, 0.9999, 0.98)

    days_per_year = st.selectbox("Days per year (utilization divisor)", [360.0, 365.0], index=0)

    pooling_mode = st.radio("Inventory strategy", ["Pooled (single pool)", "Distributed across stations"], index=0)
    n_stations = st.number_input("Number of stations", min_value=1, step=1, value=3)

    if st.button("Calculate", type="primary"):
        if pooling_mode.startswith("Pooled"):
            summary, s_total, df = summarize_pooled(
                fleet=int(fleet),
                qty_per_ac=int(qty_per_ac),
                mtbf=float(mtbf),
                annual_hours=float(annual_hours),
                tat_days=float(tat_days),
                conf=float(conf),
                days_per_year=float(days_per_year),
            )
            st.metric("Recommended spares (pooled total)", s_total)
            with st.expander("Details", expanded=True):
                st.dataframe(df)
        else:
            summary, s_total, df = summarize_distributed(
                fleet=int(fleet),
                qty_per_ac=int(qty_per_ac),
                mtbf=float(mtbf),
                annual_hours=float(annual_hours),
                tat_days=float(tat_days),
                conf=float(conf),
                days_per_year=float(days_per_year),
                n_stations=int(n_stations),
            )
            st.metric(f"Recommended spares (distributed total across {n_stations})", s_total)
            with st.expander("Per-station breakdown", expanded=True):
                st.dataframe(df)

with tab2:
    st.markdown(
        "Upload CSV/XLSX for batch processing. "
        "Columns (case-insensitive): `fleet`, `qty_per_aircraft`, `mtbf_hours`, "
        "`turn_time_days`, `avg_annual_hours`, `confidence`, `days_per_year` (optional, default 360), "
        "`strategy` (`pooled` or `distributed`), and `stations` (if distributed)."
    )
    up = st.file_uploader("Upload CSV/XLSX", type=["csv", "xlsx"])
    if up:
        df_in = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
        out_rows = []
        for _, r in df_in.iterrows():
            fleet = int(r.get("fleet"))
            qty_per_ac = int(r.get("qty_per_aircraft"))
            mtbf = float(r.get("mtbf_hours"))
            tat_days = float(r.get("turn_time_days"))
            annual_hours = float(r.get("avg_annual_hours"))
            conf = float(r.get("confidence"))
            days = float(r.get("days_per_year", 360.0))
            strategy = str(r.get("strategy", "pooled")).strip().lower()
            stations = int(r.get("stations", 1))

            if strategy == "distributed" and stations > 1:
                summary, total, df_station = summarize_distributed(
                    fleet, qty_per_ac, mtbf, annual_hours, tat_days, conf, days, stations
                )
                df_station["Batch_Row"] = _
                out_rows.append(pd.concat([pd.DataFrame([summary]), df_station], axis=1))
            else:
                summary, total, df_pool = summarize_pooled(
                    fleet, qty_per_ac, mtbf, annual_hours, tat_days, conf, days
                )
                df_pool["Batch_Row"] = _
                out_rows.append(df_pool)

        df_out = pd.concat(out_rows, ignore_index=True)
        st.success(f"Processed {len(df_in)} rows.")
        st.dataframe(df_out)

        fmt = st.selectbox("Download format", ["CSV", "Excel (XLSX)"])
        if fmt == "CSV":
            buf = io.StringIO()
            df_out.to_csv(buf, index=False)
            st.download_button("Download CSV", buf.getvalue().encode("utf-8"), file_name="provisioned_batch.csv")
        else:
            xbuf = io.BytesIO()
            with pd.ExcelWriter(xbuf, engine="openpyxl") as w:
                df_out.to_excel(w, index=False, sheet_name="ProvisionedBatch")
            st.download_button("Download XLSX", xbuf.getvalue(), file_name="provisioned_batch.xlsx")
