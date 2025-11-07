from spares_provisioning import provision_from_parameters, recommended_spares_level

# 1) Mirror the Excel tab (uses a 360-day year by default)
provision_from_parameters(
    aircraft_in_fleet=50,
    avg_annual_hours=2000,
    qty_per_aircraft=2,
    turn_time_days=25,
    mtbf_hours=40000,
    confidence_level=0.98,   # service level
)
# → {'turn_time_hours': 138.888..., 'effective_population': 100.0,
#    'lambda': 0.347222..., 'recommended_spares': 2}

# 2) Generic form if you already have turn-time in HOURS
recommended_spares_level(
    mtbf_hours=40000,
    turn_time_hours=138.8889,
    availability_target=0.98,
    population=100,          # fleet × qty_per_aircraft
)
# → (2, 0.347222...)
