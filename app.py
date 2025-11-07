import io
import pandas as pd
import streamlit as st

# --- your module (from earlier) ---
from spares_provisioning import (
    provision_from_parameters,
    provision_row,
)

st.set_page_config(page_title="Spares Provisioning (Poisson)", page_icon="✈️", layout="centered")
st.title("✈️ Spares Provisioning (Poisson)")

with st.expander("How it works", expanded=False):
    st.markdown(
        """
- Uses a Poisson model for demand during Turn Around Time (TAT).
- Sets the smallest stock level **s** so that **P{Poisson(λ) ≤ s} ≥ Confidence Level**.
- λ = (fleet × qty/aircraft) × (TAT hours / MTBF).
- TAT hours = TAT days × (annual hours / days/year). Default 360 days/year (Excel-like).
        """
    )

tab1, tab2 = st.tabs(["Single scenario", "Batch (CSV/XLSX)"])

# --- Single scenario ---
with tab1:
    col1, col2 = st.columns(2)
    with col1:
        fleet = st.number_input("Aircraft in fleet", min_value=1, value=50)
        qty_per_ac = st.number_input("Qty required per aircraft", min_value=1.0, value=2.0)
        mtbf = st.number_input("MTBF (hours)", min_value=1.0, value=40000.0, step=100.0)
    with col2:
        annual_hours = st.number_input("Average annual hours per aircraft", min_value=1.0, value=2000.0, step=50.0)
        tat_days = st.number_input("Turn Around Time (days)", min_value=0.0, value=25.0, step=1.0)
        conf = st.slider("Confidence Level (service level)", 0.50, 0.9999, 0.98)

    days_per_year = st.selectbox("Days per year (utilization divisor)", [360.0, 365.0], index=0)

    if st.button("Calculate", type="primary"):
        res = provision_from_parameters(
            aircraft_in_fleet=fleet,
            avg_annual_hours=annual_hours,
            qty_per_aircraft=qty_per_ac,
            turn_time_days=tat_days,
            mtbf_hours=mtbf,
            confidence_level=conf,
            days_per_year=float(days_per_year),
        )
        st.metric("Recommended spares", res["recommended_spares"])
        st.write(pd.DataFrame([{
            "Turn time (hours)": res["turn_time_hours"],
            "Effective population (fleet × qty/AC)": res["effective_population"],
            "Lambda (expected demand during turn)": res["lambda"],
            "Confidence Level": conf,
        }]).round(6))

# --- Batch mode ---
with tab2:
    st.markdown("Upload a CSV/XLSX with one row per scenario. Expected columns (case-insensitive):")
    st.code("mtbf_hours, turn_time_hours | turn_time_days+avg_annual_hours, availability_target, population, demand_multiplier")
    uploaded = st.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    fmt = st.selectbox("Output format", ["CSV", "Excel (XLSX)"])
    if uploaded:
        if uploaded.name.lower().endswith(".csv"):
            df = pd.read_csv(uploaded)
        else:
            # default to a first sheet; you can add a selectbox if you want
            df = pd.read_excel(uploaded)

        # Normalize a bit: if they provided days + annual hours, build turn_time_hours
        cols = {c.lower(): c for c in df.columns}
        if "turn_time_hours" not in cols and {"turn_time_days", "avg_annual_hours"}.issubset(set(cols)):
            df["turn_time_hours"] = df[cols["turn_time_days"]] * (df[cols["avg_annual_hours"]] / 360.0)

        out = df.apply(lambda r: pd.Series(provision_row(r.to_dict())), axis=1)
        st.success(f"Processed {len(out)} rows.")
        st.dataframe(out)

        if fmt == "CSV":
            buf = io.StringIO()
            out.to_csv(buf, index=False)
            st.download_button("Download CSV", buf.getvalue().encode("utf-8"), file_name="provisioned.csv")
        else:
            xbuf = io.BytesIO()
            with pd.ExcelWriter(xbuf, engine="openpyxl") as w:
                out.to_excel(w, index=False, sheet_name="Provisioned")
            st.download_button("Download XLSX", xbuf.getvalue(), file_name="provisioned.xlsx")
