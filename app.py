import json
import pandas as pd
import streamlit as st

from spares_provisioning import (
    provision_from_parameters,
    provision_row,
    provision_network,
)

st.set_page_config(page_title="Spares Provisioning (Poisson)", page_icon="✈️", layout="centered")
st.title("✈️ Spares Provisioning (Poisson)")

tab1, tab2 = st.tabs(["Single scenario", "Batch (CSV/XLSX)"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        fleet = st.number_input("Aircraft in fleet", min_value=1, value=50)
        qty_per_ac = st.number_input("Qty required per aircraft", min_value=1.0, value=2.0)
        mtbf = st.number_input("MTBF (hours)", min_value=1.0, value=40000.0, step=100.0)
    with c2:
        annual_hours = st.number_input("Average annual hours/aircraft", min_value=1.0, value=2000.0, step=50.0)
        tat_days = st.number_input("Turn Around Time (days)", min_value=0.0, value=25.0, step=1.0)
        conf = st.slider("Confidence Level", 0.50, 0.9999, 0.98)

    days_per_year = st.selectbox("Days/year for utilization", [360.0, 365.0], index=0)

    st.subheader("Network mode")
    mode = st.radio("Choose inventory strategy", ["Pooled (single pool)", "Distributed (per station)"], index=0)
    stations = st.number_input("Number of stations", min_value=1, value=1, step=1)
    weights_str = st.text_input(
        "Station weights (optional, comma-separated; e.g., 1,2,1 for uneven load)",
        value="",
        help="Leave blank for identical stations. If provided, must match #stations."
    )

    if st.button("Calculate", type="primary"):
        # Base result (Excel-like mirror)
        base = provision_from_parameters(
            aircraft_in_fleet=fleet,
            avg_annual_hours=annual_hours,
            qty_per_aircraft=qty_per_ac,
            turn_time_days=tat_days,
            mtbf_hours=mtbf,
            confidence_level=conf,
            days_per_year=float(days_per_year),
        )

        st.write("### Excel-like baseline")
        st.metric("Recommended spares (pooled baseline)", base["recommended_spares"])
        st.write(pd.DataFrame([{
            "Turn time (hours)": base["turn_time_hours"],
            "Effective population": base["effective_population"],
            "Lambda (demand during turn)": base["lambda"],
            "Confidence Level": conf,
        }]).round(6))

        # Network calculation
        selected_mode = "pooled" if mode.startswith("Pooled") else "distributed"
        weights = None
        if weights_str.strip():
            weights = [float(x.strip()) for x in weights_str.split(",") if x.strip()]

        net = provision_network(
            aircraft_in_fleet=fleet,
            avg_annual_hours=annual_hours,
            qty_per_aircraft=qty_per_ac,
            turn_time_days=tat_days,
            mtbf_hours=mtbf,
            confidence_level=conf,
            stations=int(stations),
            days_per_year=float(days_per_year),
            mode=selected_mode,
            station_weights=weights,
        )

        st.write("### Network result")
        st.write(pd.DataFrame([{
            "Stations": stations,
            "Lambda total": net["lambda_total"],
            "Pooled S_total": net["pooled"]["recommended_spares"],
            "Distributed S_total": net["distributed"]["recommended_spares_total"],
            "Pooling savings (extra units if distributed)": net["pooling_savings"],
        }]).round(6))

        if selected_mode == "distributed":
            per = pd.DataFrame({
                "Station": list(range(1, int(stations) + 1)),
                "Lambda_i": net["distributed"]["lambda_per_station"],
                "S_i": net["distributed"]["recommended_spares_per_station"],
            })
            st.write("Per-station breakdown")
            st.dataframe(per)
