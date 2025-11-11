import io
import math
import pandas as pd
import streamlit as st

# Import from your module (make sure spares_provisioning.py is in the same folder)
from spares_provisioning import (
    provision_from_parameters,
    recommended_spares_level,
)

st.set_page_config(page_title="Spares Provisioning (Poisson)", page_icon="✈️", layout="centered")
st.title("✈️ Spares Provisioning (Poisson)")

with st.expander("How it works", expanded=False):
    st.markdown(
        """
**Goal**: Choose the smallest stock level **s** so that **P{Poisson(λ) ≤ s} ≥ Confidence**.

**Key formulas**
- Turn time (hours) = **TAT_days × (annual_hours / days_per_year)** (default 360 = Excel-like)
- Effective population = **fleet × qty_per_aircraft**
- **λ** (expected demand during turn) = **effective_population × (turn_time_hours / MTBF)**

**Stations policy**
- **Pooled (central stock):** compute one λ using the total population; get one **s_total**. If you must place parts across stations, split **s_total** after the fact (shown below).
- **Distributed (per-station):** split the population evenly across N stations, compute a per-station λ and **s_station**, then **total = N × s_station**.
        """
    )

tab1, tab2 = st.tabs(["Single scenario", "Batch (CSV/XLSX)"])

# ------------------------
# Helpers
# ------------------------
def split_even(n_items: int, n_bins: int):
    """Return a list of length n_bins that splits n_items as evenly as possible."""
    base = n_items // n_bins
    rem = n_items % n_bins
    return [base + 1 if i < rem else base for i in range(n_bins)]

def summarize_pooled(fleet, qty_per_ac, tat_days, annual_hours, days_per_year, mtbf, conf, n_stations):
    # Use module function that mirrors the Excel convention
    res = provision_from_parameters(
        aircraft_in_fleet=fleet,
        avg_annual_hours=annual_hours,
        qty_per_aircraft=qty_per_ac,
        turn_time_days=tat_days,
        mtbf_hours=mtbf,
        confidence_level=conf,
        days_per_year=float(days_per_year),
    )
    s_total = int(res["recommended_spares"])
    # If stations > 1, produce a suggested split
    per_station = split_even(s_total, n_stations) if n_stations > 1 else [s_total]
    df_dist = pd.DataFrame({
        "Station": [f"Station {i+1}" for i in range(n_stations)],
        "Allocated spares": per_station,
    }) if n_stations > 1 else pd.DataFrame({"Station": ["Central"], "Allocated spares": [s_total]})
    return res, s_total, df_dist

def summarize_distributed(fleet, qty_per_ac, tat_days, annual_hours, days_per_year, mtbf, conf, n_stations):
    # Compute turn_time_hours once
    turn_time_hours = tat_days * (annual_hours / float(days_per_year))
    total_pop = fleet * qty_per_ac
    # Split population as evenly as possible across stations (integers if possible)
    # Population is a count of installed units: make it integer
    total_pop_int = int(total_pop)
    pops = split_even(total_pop_int, n_stations) if total_pop_int >= n_stations else [1] * total_pop_int + [0] * (n_stations - total_pop_int)
    # For any fractional remainder (if user inputs led to non-integer, we already forced qty_per_ac to int, so safe)

    per_station_rows = []
    s_station_list = []
    lam_list = []
    for i, pop_i in enumerate(pops, start=1):
        if pop_i <= 0:
            s_i, lam_i = 0, 0.0
        else:
            s_i, lam_i = recommended_spares_level(
                mtbf_hours=mtbf,
                turn_time_hours=turn_time_hours,
                availability_target=conf,
                population=pop_i,
                demand_multiplier=1.0,
            )
        per_station_rows.append({"Station": f"Station {i}", "Population": pop_i, "Lambda": lam_i, "Recommended spares": int(s_i)})
        s_station_list.append(int(s_i))
        lam_list.append(lam_i)

    total_s = int(sum(s_station_list))
    # Provide a compact summary similar to provision_from_parameters output
    summary = {
        "turn_time_hours": float(turn_time_hours),
        "effective_population": float(total_pop_int),
        "lambda": float(sum(lam_list)),  # not directly used for stocking in distributed mode, but informative
        "recommended_spares": total_s,
    }
    return summary, total_s, pd.DataFrame(per_station_rows)

# ------------------------
# Single scenario UI
# ------------------------
with tab1:
    colL, colR = st.columns(2)
    with colL:
        fleet = st.number_input("Aircraft in fleet", min_value=1, step=1, value=50, format="%d")
        qty_per_ac = st.number_input("Qty required per aircraft", min_value=1, step=1, value=2, format="%d")  # integer step
        mtbf = st.number_input("MTBF (hours)", min_value=1.0, value=40000.0, step=100.0)
        stations = st.number_input("Number of stations", min_value=1, step=1, value=1, format="%d")
    with colR:
        annual_hours = st.number_input("Average annual hours per aircraft", min_value=1.0, value=2000.0, step=50.0)
        tat_days = st.number_input("Turn Around Time (days)", min_value=0.0, value=25.0, step=1.0)
        conf = st.slider("Confidence Level (service level)", 0.50, 0.9999, 0.98)
        days_per_year = st.selectbox("Days per year (utilization divisor)", [360.0, 365.0], index=0)

    policy = st.radio("Stations policy", ["Pooled (central stock)", "Distributed (per-station)"], index=0, horizontal=False)

    if st.button("Calculate", type="primary"):
        if policy.startswith("Pooled"):
            summary, s_total, df_dist = summarize_pooled(
                fleet, qty_per_ac, tat_days, annual_hours, days_per_year, mtbf, conf, stations
            )
            st.subheader("Result (Pooled)")
            st.metric("Recommended spares (total)", s_total)
            st.write(pd.DataFrame([{
                "Turn time (hours)": summary["turn_time_hours"],
                "Effective population (fleet × qty/AC)": summary["effective_population"],
                "Lambda (expected demand during turn)": summary["lambda"],
                "Confidence Level": conf,
            }]).round(6))
            st.caption("Suggested placement across stations (post-allocation of central pool):")
            st.dataframe(df_dist, use_container_width=True)

        else:
            summary, s_total, df_per_station = summarize_distributed(
                fleet, qty_per_ac, tat_days, annual_hours, days_per_year, mtbf, conf, stations
            )
            st.subheader("Result (Distributed)")
            st.metric("Recommended spares (sum over stations)", s_total)
            st.write(pd.DataFrame([{
                "Turn time (hours)": summary["turn_time_hours"],
                "Effective population (sum across stations)": summary["effective_population"],
                "Confidence Level": conf,
            }]).round(6))
            st.caption("Per-station populations, λ, and stock:")
            st.dataframe(df_per_station, use_container_width=True)

# ------------------------
# Batch mode
# ------------------------
with tab2:
    st.markdown(
        """
Upload CSV/XLSX with one row per scenario.

**Required (case-insensitive):**
- `fleet`, `qty_per_aircraft`, `mtbf_hours`, `turn_time_days`, `avg_annual_hours`, `confidence_level`

**Stations options (optional):**
- `stations` (default 1)
- `policy` = `"pooled"` or `"distributed"` (default `"pooled"`)
- `days_per_year` (default 360)

**Output columns:**
- Recommended spares (total); and if distributed, a JSON-like per-station summary column.
        """
    )
    up = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    fmt = st.selectbox("Output format", ["CSV", "Excel (XLSX)"])

    if up:
        df_in = pd.read_csv(up) if up.name.lower().endswith(".csv") else pd.read_excel(up)
        cols = {c.lower(): c for c in df_in.columns}

        def get_col(name, default=None, required=False):
            key = cols.get(name.lower())
            if key is None:
                if required:
                    raise KeyError(f"Missing required column: {name}")
                return default
            return key

        try:
            out_rows = []
            for _, r in df_in.iterrows():
                fleet = int(r[get_col("fleet", required=True)])
                qty_per_ac = int(r[get_col("qty_per_aircraft", required=True)])
                mtbf = float(r[get_col("mtbf_hours", required=True)])
                tat_days = float(r[get_col("turn_time_days", required=True)])
                annual_hours = float(r[get_col("avg_annual_hours", required=True)])
                conf = float(r[get_col("confidence_level", required=True)])
                stations = int(r.get(get_col("stations", default="stations") or "stations", 1))
                policy = str(r.get(get_col("policy", default="policy") or "policy", "pooled")).strip().lower()
                days_per_year = float(r.get(get_col("days_per_year", default="days_per_year") or "days_per_year", 360.0))

                if policy not in ("pooled", "distributed"):
                    policy = "pooled"

                if policy == "pooled":
                    summary, s_total, df_dist = summarize_pooled(
                        fleet, qty_per_ac, tat_days, annual_hours, days_per_year, mtbf, conf, stations
                    )
                    per_station_alloc = df_dist.to_dict(orient="records")
                    out_rows.append({
                        "policy": "pooled",
                        "stations": stations,
                        "recommended_spares_total": int(s_total),
                        "turn_time_hours": summary["turn_time_hours"],
                        "effective_population": summary["effective_population"],
                        "lambda": summary["lambda"],
                        "confidence_level": conf,
                        "allocation": per_station_alloc,
                    })
                else:
                    summary, s_total, df_ps = summarize_distributed(
                        fleet, qty_per_ac, tat_days, annual_hours, days_per_year, mtbf, conf, stations
                    )
                    per_station = df_ps.to_dict(orient="records")
                    out_rows.append({
                        "policy": "distributed",
                        "stations": stations,
                        "recommended_spares_total": int(s_total),
                        "turn_time_hours": summary["turn_time_hours"],
                        "effective_population": summary["effective_population"],
                        "confidence_level": conf,
                        "per_station": per_station,
                    })

            out_df = pd.DataFrame(out_rows)
            st.success(f"Processed {len(out_df)} scenarios.")
            st.dataframe(out_df, use_container_width=True)

            if fmt == "CSV":
                buf = io.StringIO()
                out_df.to_csv(buf, index=False)
                st.download_button("Download CSV", buf.getvalue().encode("utf-8"), file_name="provisioned_scenarios.csv")
            else:
                xbuf = io.BytesIO()
                with pd.ExcelWriter(xbuf, engine="openpyxl") as w:
                    out_df.to_excel(w, index=False, sheet_name="Provisioned")
                st.download_button("Download XLSX", xbuf.getvalue(), file_name="provisioned_scenarios.xlsx")

        except KeyError as e:
            st.error(str(e))
        except Exception as e:
            st.error(f"Error while processing: {e}")
