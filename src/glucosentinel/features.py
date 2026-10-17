"""Feature engineering: fuses the static EHR stream, the dynamic wearable stream, the patient's
own routine and the digital twin's physiological state into one row every 5 minutes.

Feature groups (used for the ablation study):
  E  EHR / FHIR record          (who the patient is)
  W  wearables + CGM + meal log (what is happening now)
  R  personal routine           (what usually happens at this time of day, learned in week 1)
  T  digital twin               (what the calibrated physiology says will happen next)
"""
import numpy as np
import pandas as pd

from .twin import PatientTwin, build_inputs, H, DT

SPIKE_MGDL = 180.0
HORIZON_MIN = 120
CAL_STEPS = 7 * 288          # onboarding week used to calibrate the twin and learn routine

EHR_FEATURES = ["age", "male", "bmi", "waist_cm", "diabetes_duration_y", "hypertension", "dyslipidaemia",
                "family_history_t2d", "tcf7l2_rs7903146_risk_alleles", "t2d_prs", "on_metformin",
                "on_sulfonylurea", "on_dpp4i", "sbp", "ldl", "hdl", "triglycerides", "egfr",
                "hba1c_pct", "hba1c_trend_12m", "fpg_mgdl", "resting_hr", "prediabetes"]


def spike_episodes(cgm, min_len=3):
    """Onset indices of excursions above 180 mg/dL lasting at least 15 minutes."""
    x = pd.Series(cgm).interpolate(limit=3, limit_area="inside").to_numpy()
    above = np.nan_to_num(x, nan=0.0) > SPIKE_MGDL
    onsets, i, n = [], 0, len(x)
    while i < n:
        if above[i] and (i == 0 or not above[i - 1]):
            j = i
            while j < n and above[j]:
                j += 1
            if j - i >= min_len:
                onsets.append(i)
            i = j
        else:
            i += 1
    return np.array(onsets, dtype=int), x


def labels(cgm):
    onsets, x = spike_episodes(cgm)
    n = len(cgm)
    y = np.zeros(n, dtype=np.int8)
    for o in onsets:
        y[max(o - H, 0):o] = 1                      # onset falls inside the next 2 hours
    fut_ok = pd.Series(~np.isnan(x)).rolling(H, min_periods=1).sum().shift(-H).to_numpy()
    eligible = (~np.isnan(cgm)) & (np.nan_to_num(cgm, nan=999) <= SPIKE_MGDL) & (np.nan_to_num(fut_ok) >= 18)
    return y, eligible, onsets


def _roll(s, w, f):
    return getattr(s.rolling(w, min_periods=1), f)()


def wearable_features(t, meals):
    """t: one patient's 5-min frame indexed by timestamp."""
    f = pd.DataFrame(index=t.index)
    g = t.cgm_mgdl.interpolate(limit=3, limit_area="inside")
    gf = g.ffill()
    f["cgm_now"] = gf
    for k in (1, 3, 6, 12):
        f[f"cgm_delta_{k * 5}m"] = gf - gf.shift(k)
    f["cgm_slope_30m"] = (gf - gf.shift(6)) / 30.0
    f["cgm_accel"] = (gf - gf.shift(3)) - (gf.shift(3) - gf.shift(6))
    for w, name in ((12, "1h"), (36, "3h"), (288, "24h")):
        f[f"cgm_mean_{name}"] = _roll(gf, w, "mean")
        f[f"cgm_std_{name}"] = _roll(gf, w, "std")
    f["cgm_max_3h"] = _roll(gf, 36, "max")
    f["cgm_min_3h"] = _roll(gf, 36, "min")
    f["tar180_24h"] = _roll((gf > SPIKE_MGDL).astype(float), 288, "mean")
    hi = (gf > SPIKE_MGDL).to_numpy()
    last, since = -10_000, np.zeros(len(gf))
    for i, v in enumerate(hi):
        if v:
            last = i
        since[i] = min((i - last) * 5, 1440)
    f["min_since_above180"] = since

    # meal log
    carbs = pd.Series(0.0, index=t.index)
    gl = pd.Series(0.0, index=t.index)
    mi = t.index.get_indexer(meals.timestamp.dt.floor("5min"))
    ok = mi >= 0
    np.add.at(carbs.values, mi[ok], meals.carbs_logged_g.to_numpy()[ok])
    np.add.at(gl.values, mi[ok], (meals.carbs_logged_g * meals.gi / 100).to_numpy()[ok])
    for w in (6, 12, 24, 48):
        f[f"carbs_logged_{w * 5}m"] = _roll(carbs, w, "sum")
    f["glycaemic_load_2h"] = _roll(gl, 24, "sum")
    lastm, sm = -10_000, np.zeros(len(carbs))
    for i, v in enumerate(carbs.to_numpy() > 0):
        if v:
            lastm = i
        sm[i] = min((i - lastm) * 5, 720)
    f["min_since_logged_meal"] = sm

    # activity
    st = t.steps.fillna(0)
    for w in (3, 6, 12, 24):
        f[f"steps_{w * 5}m"] = _roll(st, w, "sum")
    walking = (st >= 400).to_numpy()
    lw, sw = -10_000, np.zeros(len(st))
    for i, v in enumerate(walking):
        if v:
            lw = i
        sw[i] = min((i - lw) * 5, 720)
    f["min_since_walk"] = sw
    f["steps_today"] = st.groupby(t.index.normalize()).cumsum()

    # heart
    hr = t.heart_rate_bpm
    f["hr_15m"] = _roll(hr, 3, "mean")
    f["hr_1h"] = _roll(hr, 12, "mean")
    hr_rest = hr.rolling(288, min_periods=12).quantile(0.1)
    f["hr_above_rest"] = f["hr_15m"] - hr_rest
    rm = t.hrv_rmssd_ms.ffill(limit=36)
    f["hrv_rmssd_3h"] = _roll(rm, 36, "mean")
    base = t.hrv_rmssd_ms.rolling(288 * 3, min_periods=48).median()
    f["hrv_rel_baseline"] = f["hrv_rmssd_3h"] / base

    # sleep (night before the current day)
    stg = t.sleep_stage.astype(str)
    night = (t.index + pd.Timedelta(hours=6)).normalize()
    sdf = pd.DataFrame({"d": night, "sl": stg.isin(["light", "deep", "rem"]).to_numpy(),
                        "dp": (stg == "deep").to_numpy(), "aw": (stg == "awake").to_numpy()})
    agg = sdf.groupby("d").agg(sl=("sl", "sum"), dp=("dp", "sum"), aw=("aw", "sum"))
    day = t.index.normalize()
    f["sleep_hours_last_night"] = agg.sl.reindex(day).to_numpy() * 5 / 60
    f["awake_in_bed_min_last_night"] = agg.aw.reindex(day).to_numpy() * 5
    f["deep_sleep_frac_last_night"] = (agg.dp / agg.sl.clip(lower=1)).reindex(day).to_numpy()

    hours = t.index.hour + t.index.minute / 60
    f["hour_sin"] = np.sin(2 * np.pi * hours / 24)
    f["hour_cos"] = np.cos(2 * np.pi * hours / 24)
    f["weekend"] = (t.index.dayofweek >= 5).astype(int)
    return f, carbs


def routine_features(f, carbs, onsets, n):
    """Personal routine learned only from the onboarding week (no peeking at later days)."""
    slot = np.arange(n) % 288
    cal = slice(0, CAL_STEPS)
    c = carbs.to_numpy()
    fut_carbs = pd.Series(c[::-1]).rolling(H, min_periods=1).sum().to_numpy()[::-1]   # carbs in [t, t+2h)
    ons = np.zeros(n); ons[onsets[onsets < n]] = 1
    fut_spk = (pd.Series(ons[::-1]).rolling(H, min_periods=1).sum().to_numpy()[::-1] > 0).astype(float)
    d1 = pd.DataFrame({"slot": slot[cal], "carbs": fut_carbs[cal], "spk": fut_spk[cal]}).groupby("slot").mean()
    # carb-weighted typical offset of the next meal
    off = np.zeros(288)
    cday = c[:CAL_STEPS].reshape(7, 288).mean(0)
    for s in range(288):
        w = np.array([cday[(s + k) % 288] for k in range(H)])
        off[s] = (np.arange(H) * w).sum() / w.sum() if w.sum() > 0 else H
    r = pd.DataFrame(index=f.index)
    r["habit_carbs_next_2h"] = d1.carbs.reindex(slot).to_numpy()
    r["habit_spike_rate_next_2h"] = d1.spk.reindex(slot).to_numpy()
    r["habit_meal_offset_min"] = off[slot] * DT
    g = f["cgm_now"].to_numpy()[:CAL_STEPS]
    r["week1_mean_glucose"] = np.nanmean(g)
    r["week1_tar180"] = np.nanmean(g > SPIKE_MGDL)
    return r, off[slot].astype(int)


def twin_features(twin, inp, habit_carbs, habit_offset):
    st = twin.filter(inp)
    fc = twin.forecast(st, inp)
    fc_h = twin.forecast(st, inp, extra_carbs=habit_carbs * 1000, extra_offset=habit_offset)
    T = pd.DataFrame(index=inp["idx"])
    T["twin_fc_30m"] = fc[:, 5]
    T["twin_fc_60m"] = fc[:, 11]
    T["twin_fc_120m"] = fc[:, 23]
    T["twin_max_2h"] = fc.max(1)
    T["twin_habit_max_2h"] = fc_h.max(1)
    T["twin_habit_minutes_above180"] = (fc_h > SPIKE_MGDL).sum(1) * DT
    T["twin_carbs_on_board_g"] = (st[3] + st[4] + st[5] + st[6]) / 1000.0
    T["twin_insulin_action"] = st[1] * 1000
    p, pr = twin.params, twin.prior
    T["twin_insulin_sensitivity"] = p["SI"] * 1e4
    T["twin_beta_cell_response"] = p["gamma"] * 100
    T["twin_basal_glucose"] = p["Gb"]
    T["twin_dawn_effect"] = p["dawn"]
    T["twin_si_vs_ehr_prior"] = p["SI"] / pr["SI"]
    return T, st, fc


def ehr_row(ehr):
    e = dict(ehr)
    e["male"] = int(e.get("sex") == "M")
    e["prediabetes"] = int(e.get("icd10") == "R73.03")
    return {k: float(e[k]) for k in EHR_FEATURES}


def build_patient(pid, ts_p, meals_p, ehr, calibrate=True):
    """Everything for one patient: feature frame, labels, calibrated twin and its states."""
    t = ts_p.set_index("timestamp").sort_index()
    n = len(t)
    inp = build_inputs(ts_p, meals_p, t.index[0], n)
    twin = PatientTwin(ehr)
    if calibrate:
        twin.calibrate(inp, n_days=7)
    w, carbs = wearable_features(t, meals_p)
    y, eligible, onsets = labels(t.cgm_mgdl.to_numpy())
    r, off = routine_features(w, carbs, onsets, n)
    tw, states, fc = twin_features(twin, inp, r["habit_carbs_next_2h"].fillna(0).to_numpy(), off)
    e = pd.DataFrame([ehr_row(ehr)] * n, index=t.index)
    df = pd.concat([w, e, r, tw], axis=1)
    df["patient_id"] = pid
    df["y_spike_2h"] = y
    df["eligible"] = eligible
    df["eval_week"] = (np.arange(n) >= CAL_STEPS)
    gf = w["cgm_now"].to_numpy()
    for k in (6, 12, 18, 24):
        fut = np.full(n, np.nan); fut[:-k] = t.cgm_mgdl.to_numpy()[k:]
        df[f"target_cgm_{k * 5}m"] = fut
    return df.reset_index().rename(columns={"index": "timestamp"}), twin, states, onsets


GROUPS = None


def feature_groups(columns):
    W = [c for c in columns if c.startswith(("cgm_", "tar180", "min_since", "carbs_logged", "glycaemic",
                                             "steps_", "hr_", "hrv_", "sleep_", "awake_in_bed", "deep_sleep",
                                             "hour_", "weekend")) and c != "cgm_raw"]
    E = list(EHR_FEATURES)
    R = [c for c in columns if c.startswith(("habit_", "week1_"))]
    T = [c for c in columns if c.startswith("twin_")]
    return {"E": E, "W": W, "R": R, "T": T}
