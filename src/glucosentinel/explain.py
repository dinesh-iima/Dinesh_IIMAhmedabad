"""Doctor-friendly names for model features, used by the dashboard and the reports."""

FRIENDLY = {
    "cgm_now": "Current glucose", "cgm_delta_5m": "Glucose change, last 5 min",
    "cgm_delta_15m": "Glucose change, last 15 min", "cgm_delta_30m": "Glucose change, last 30 min",
    "cgm_delta_60m": "Glucose change, last 60 min", "cgm_slope_30m": "Glucose trend (mg/dL per min)",
    "cgm_accel": "Glucose acceleration", "cgm_mean_1h": "Mean glucose, last hour",
    "cgm_std_1h": "Glucose variability, last hour", "cgm_mean_3h": "Mean glucose, last 3 h",
    "cgm_std_3h": "Glucose variability, last 3 h", "cgm_mean_24h": "Mean glucose, last 24 h",
    "cgm_std_24h": "Glucose variability, last 24 h", "cgm_max_3h": "Peak glucose, last 3 h",
    "cgm_min_3h": "Lowest glucose, last 3 h", "tar180_24h": "Time above 180, last 24 h",
    "min_since_above180": "Minutes since last spike",
    "carbs_logged_30m": "Carbs logged, last 30 min", "carbs_logged_60m": "Carbs logged, last hour",
    "carbs_logged_120m": "Carbs logged, last 2 h", "carbs_logged_240m": "Carbs logged, last 4 h",
    "glycaemic_load_2h": "Glycaemic load, last 2 h", "min_since_logged_meal": "Minutes since last meal",
    "steps_15m": "Steps, last 15 min", "steps_30m": "Steps, last 30 min", "steps_60m": "Steps, last hour",
    "steps_120m": "Steps, last 2 h", "min_since_walk": "Minutes since last walk", "steps_today": "Steps today",
    "hr_15m": "Heart rate, last 15 min", "hr_1h": "Heart rate, last hour", "hr_above_rest": "Heart rate above resting",
    "hrv_rmssd_3h": "HRV (RMSSD), last 3 h", "hrv_rel_baseline": "HRV vs personal baseline",
    "sleep_hours_last_night": "Sleep last night (h)", "awake_in_bed_min_last_night": "Awake in bed last night (min)",
    "deep_sleep_frac_last_night": "Deep sleep share last night",
    "hour_sin": "Time of day", "hour_cos": "Time of day", "weekend": "Weekend",
    "age": "Age", "male": "Sex", "bmi": "BMI", "waist_cm": "Waist circumference",
    "diabetes_duration_y": "Years since diagnosis", "hypertension": "Hypertension", "dyslipidaemia": "Dyslipidaemia",
    "family_history_t2d": "Family history of T2D", "tcf7l2_rs7903146_risk_alleles": "TCF7L2 risk alleles",
    "t2d_prs": "Polygenic risk score", "on_metformin": "On metformin", "on_sulfonylurea": "On sulfonylurea",
    "on_dpp4i": "On DPP-4 inhibitor", "sbp": "Systolic BP", "ldl": "LDL cholesterol", "hdl": "HDL cholesterol",
    "triglycerides": "Triglycerides", "egfr": "eGFR", "hba1c_pct": "HbA1c", "hba1c_trend_12m": "HbA1c trend, 12 months",
    "fpg_mgdl": "Fasting glucose (lab)", "resting_hr": "Resting heart rate (EHR)", "prediabetes": "Prediabetes",
    "habit_carbs_next_2h": "Usual carbs in the next 2 h", "habit_spike_rate_next_2h": "Usual spike rate at this hour",
    "habit_meal_offset_min": "Usual time to next meal", "week1_mean_glucose": "Personal mean glucose",
    "week1_tar180": "Personal time above 180",
    "twin_fc_30m": "Twin forecast, +30 min", "twin_fc_60m": "Twin forecast, +60 min",
    "twin_fc_120m": "Twin forecast, +120 min", "twin_max_2h": "Twin peak, next 2 h",
    "twin_habit_max_2h": "Twin peak if usual meal is eaten", "twin_habit_minutes_above180": "Twin minutes above 180 if usual meal",
    "twin_carbs_on_board_g": "Carbs still being absorbed", "twin_insulin_action": "Active insulin action",
    "twin_insulin_sensitivity": "Insulin sensitivity (twin)", "twin_beta_cell_response": "Beta-cell response (twin)",
    "twin_basal_glucose": "Basal glucose (twin)", "twin_dawn_effect": "Dawn effect (twin)",
    "twin_si_vs_ehr_prior": "Sensitivity vs EHR expectation",
}

GROUP_NAME = {"E": "EHR (static)", "W": "Wearables + CGM (dynamic)", "R": "Personal routine", "T": "Digital twin physiology"}


def friendly(c):
    return FRIENDLY.get(c, c.replace("_", " "))
