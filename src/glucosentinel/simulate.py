"""Generates the two data streams the challenge asks for.

Static stream : EHR table + FHIR R4 bundles (see fhir.py).
Dynamic stream: 5-minute wearable and CGM data that mimics an Apple Health / Google Fit /
                Libre-style export: CGM glucose, step count, heart rate, HRV (RMSSD),
                sleep stage, plus a phone-app meal log.

Ground truth comes from running the physiology model at 1-minute resolution with each
person's hidden parameters and day-by-day behaviour. The model and twin only ever see
the noisy, lagged, partly-missing sensor and log data.
"""
import numpy as np
import pandas as pd

from . import physiology as phy
from .foods import FOODS, menu_for, fast_fraction

START = pd.Timestamp("2026-08-31 00:00")   # day 0 is a warm-up day and is discarded
N_DAYS = 15
MIN_PER_DAY = 1440


def _minute(day, hour):
    return int(round(day * MIN_PER_DAY + hour * 60))


def _behaviour(p, rng, n_days):
    """Build minute-level behaviour arrays for one person."""
    life = p["_life"]
    T = n_days * MIN_PER_DAY
    steps = np.zeros(T)
    stage = np.full(T, -1, dtype=np.int8)          # -1 out of bed, 0 awake in bed, 1 light, 2 deep, 3 REM
    stress = np.zeros(T)
    meals, days = [], []
    festival_day = int(rng.integers(2, n_days)) if rng.random() < 0.4 else -1

    for d in range(n_days):
        dow = (START + pd.Timedelta(days=d)).dayofweek
        weekend = dow >= 5
        # ---------------- previous night's sleep ----------------
        bed = life["t_bed"] + rng.normal(0, 0.35) - 24.0           # relative to day d midnight
        wake = life["t_wake"] + rng.normal(0, 0.25) + (0.8 if weekend else 0.0)
        poor = rng.random() < life["poor_sleep_prop"]
        awakenings = int(rng.integers(5, 10)) if poor else int(rng.integers(0, 3))
        if poor:
            if rng.random() < 0.6:
                bed += rng.uniform(0.8, 2.2)
            else:
                wake -= rng.uniform(0.5, 1.3)
        duration = wake - bed
        b0, w0 = max(_minute(d, bed), 0), _minute(d, wake)
        n_sleep = max(w0 - b0, 0)
        if n_sleep > 0:
            m = np.arange(n_sleep)
            frac = m / max(n_sleep, 1)
            cyc = (m % 90) / 90.0
            st = np.ones(n_sleep, dtype=np.int8)
            deep_end = 0.15 + 0.35 * (1 - frac) * (0.55 if poor else 1.0)
            st[(cyc >= 0.15) & (cyc < deep_end)] = 2
            st[cyc >= 1 - (0.12 + 0.18 * frac)] = 3
            for _ in range(awakenings):
                a = int(rng.integers(0, n_sleep))
                st[a:a + int(rng.integers(4, 18))] = 0
            stage[b0:b0 + n_sleep] = st
        poor_score = float(np.clip((7.0 - duration) / 3.0, 0, 1) * 0.6 + np.clip((awakenings - 1) / 8.0, 0, 1) * 0.4)

        # ---------------- stress ----------------
        stress_day = rng.random() < life["stress_prop"] * (0.6 if weekend else 1.15)
        if stress_day:
            stress[_minute(d, 9.0):_minute(d, 19.5)] = rng.uniform(0.6, 1.0)

        # ---------------- meals ----------------
        shift = 0.6 if weekend else 0.0
        plan = [("breakfast", life["t_breakfast"] + shift, 0.95),
                ("lunch", life["t_lunch"] + shift * 0.5, 0.98),
                ("snack", life["t_snack"], life["snack_prob"]),
                ("dinner", life["t_dinner"], 0.97)]
        day_meals = []
        for slot, hr, prob in plan:
            if rng.random() > prob:
                continue
            names, w = menu_for(slot, life["region"])
            dish = rng.choice(names, p=np.array(w) / np.sum(w))
            if slot == "snack" and rng.random() < life["sweet_freq"]:
                dish = "Sweet (gulab jamun or jalebi)"
            t = hr + rng.normal(0, 0.33)
            day_meals.append((slot, t, dish))
        if d == festival_day:
            day_meals.append(("snack", life["t_lunch"] + 0.6, "Sweet (gulab jamun or jalebi)"))
            day_meals.append(("snack", life["t_dinner"] + 0.5, "Sweet (gulab jamun or jalebi)"))
        for slot, t, dish in day_meals:
            f = FOODS[dish]
            carbs = f["carbs"] * life["portion"] * np.exp(rng.normal(0, 0.12))
            meals.append(dict(minute=_minute(d, t), day=d, slot=slot, dish=dish, carbs_g=carbs, gi=f["gi"]))

        # ---------------- activity ----------------
        wake_m, bed_m = _minute(d, wake), _minute(d, life["t_bed"] + rng.normal(0, 0.3))
        hours_awake = max((bed_m - wake_m) / 60.0, 1)
        n_bouts = rng.poisson(2.6 * life["activity_level"] * hours_awake * (0.75 if weekend else 1.0))
        for _ in range(n_bouts):
            s0 = int(rng.integers(wake_m, max(bed_m - 5, wake_m + 1)))
            steps[s0:s0 + int(rng.integers(1, 9))] = rng.uniform(50, 105)
        if rng.random() < life["morning_walk"]:
            s0 = wake_m + int(rng.integers(10, 60))
            steps[s0:s0 + int(rng.integers(20, 46))] = rng.uniform(95, 118)
        for slot, t, dish in day_meals:
            if slot in ("lunch", "dinner") and rng.random() < life["walk_after_meal"]:
                s0 = _minute(d, t) + int(rng.integers(5, 26))
                steps[s0:s0 + int(rng.integers(10, 26))] = rng.uniform(85, 108)
        steps[stage >= 0] = 0.0
        days.append(dict(day=d, weekend=weekend, poor_sleep=poor, poor_score=poor_score,
                         sleep_hours=duration, awakenings=awakenings, stress_day=stress_day,
                         festival=(d == festival_day)))
    steps[stage >= 0] = 0.0
    return steps, stage, stress, meals, days


def simulate_cohort(cohort, seed=11):
    rng = np.random.default_rng(seed)
    N = len(cohort)
    T = N_DAYS * MIN_PER_DAY
    carb_f = np.zeros((N, T)); carb_s = np.zeros((N, T))
    act = np.zeros((N, T)); gb = np.zeros((N, T)); sid = np.zeros((N, T))
    stage = np.zeros((N, T), dtype=np.int8); stress = np.zeros((N, T))
    meal_rows, day_rows = [], []
    hours = (np.arange(T) % MIN_PER_DAY) / 60.0

    for i, (_, p) in enumerate(cohort.iterrows()):
        prm = p["_params"]
        st, sg, sr, meals, days = _behaviour(p, rng, N_DAYS)
        act[i], stage[i], stress[i] = st, sg, sr
        for m in meals:
            ff = fast_fraction(m["gi"])
            if 0 <= m["minute"] < T:
                carb_f[i, m["minute"]] += m["carbs_g"] * 1000 * ff
                carb_s[i, m["minute"]] += m["carbs_g"] * 1000 * (1 - ff)
            m["patient_id"] = p["patient_id"]
            meal_rows.append(m)
        day_offset = rng.normal(0, 3.0, N_DAYS)
        si_day = np.ones(N_DAYS)
        for dd in days:
            si_day[dd["day"]] = 1 - 0.26 * dd["poor_score"] - 0.08 * dd["stress_day"] + rng.normal(0, 0.04)
            dd["patient_id"] = p["patient_id"]
            day_rows.append(dd)
        dayidx = np.arange(T) // MIN_PER_DAY
        gb[i] = (prm["Gb"] + prm["dawn"] * np.exp(-0.5 * ((hours - 6.3) / 1.1) ** 2)
                 + 10.0 * stress[i] + day_offset[dayidx])
        sid[i] = si_day[dayidx]

    P = {k: np.array([c[k] for c in cohort["_params"]]) for k in ("V", "SG", "SI", "p2", "n", "gamma", "Ib", "ex")}
    s = phy.init_state(gb[:, 0], P["Ib"])
    G = np.zeros((N, T)); Gi = np.zeros((N, T))
    a_eff = np.zeros(N)
    for t in range(T):
        a_eff += ((act[:, t] >= 80).astype(float) - a_eff) / 120.0
        s = phy.step(s, P, carb_f[:, t], carb_s[:, t], act[:, t], gb[:, t], sid[:, t] * (1 + 0.25 * a_eff), 1.0)
        G[:, t] = s["G"]; Gi[:, t] = s["Gi"]

    # ---------------- wearable signals ----------------
    rest = cohort["resting_hr"].to_numpy()[:, None]
    hr = (rest + 0.28 * act + 7 * stress - 7 * (stage >= 1) + 3 * (stage == 3)
          + 3 * np.sin(2 * np.pi * (hours - 15) / 24) + rng.normal(0, 2.5, (N, T)))

    frames = []
    keep = slice(MIN_PER_DAY, T)              # drop warm-up day
    idx5 = np.arange(MIN_PER_DAY, T, 5)
    ts = START + pd.to_timedelta(idx5, unit="min")
    poor_by_day = {(r["patient_id"], r["day"]): r["poor_score"] for r in day_rows}
    for i, (_, p) in enumerate(cohort.iterrows()):
        # CGM: interstitial glucose + autocorrelated noise + 2 percent dropouts
        gi5 = Gi[i, idx5]
        e = np.zeros(len(idx5))
        for k in range(1, len(e)):
            e[k] = 0.7 * e[k - 1] + rng.normal(0, 3.5)
        cgm = gi5 * (1 + rng.normal(0, 0.02, len(idx5))) + e
        cgm = np.clip(np.round(cgm), 40, 400).astype(float)
        drop = rng.random(len(idx5)) < 0.02
        for _ in range(rng.integers(1, 4)):       # a few short signal-loss gaps
            g0 = int(rng.integers(0, len(idx5) - 12))
            drop[g0:g0 + int(rng.integers(3, 12))] = True
        cgm[drop] = np.nan
        steps5 = act[i, :].reshape(-1, 5).sum(1)[MIN_PER_DAY // 5:]
        hr5 = hr[i, :].reshape(-1, 5).mean(1)[MIN_PER_DAY // 5:]
        stage5 = stage[i, idx5 + 2]
        stress5 = stress[i, :].reshape(-1, 5).mean(1)[MIN_PER_DAY // 5:]
        dayix = idx5 // MIN_PER_DAY
        poor5 = np.array([poor_by_day[(p["patient_id"], d)] for d in dayix])
        rmssd = (p["baseline_rmssd"] * (1 - 0.30 * stress5) * (1 - 0.18 * poor5)
                 * np.where(stage5 == 2, 1.25, 1.0) * np.exp(rng.normal(0, 0.15, len(idx5))))
        rmssd[steps5 > 40] = np.nan                  # HRV is only reported at rest
        frames.append(pd.DataFrame({
            "patient_id": p["patient_id"], "timestamp": ts,
            "cgm_mgdl": cgm, "steps": np.round(steps5).astype(int),
            "heart_rate_bpm": np.round(hr5, 1), "hrv_rmssd_ms": np.round(rmssd, 1),
            "sleep_stage": pd.Categorical.from_codes(np.where(stage5 < 0, 4, stage5),
                                                     ["awake", "light", "deep", "rem", "out_of_bed"]),
            "_true_glucose": np.round(G[i, idx5], 1),
        }))
    ts_df = pd.concat(frames, ignore_index=True)

    meals = pd.DataFrame(meal_rows)
    meals = meals[(meals.minute >= MIN_PER_DAY) & (meals.minute < T)].copy()
    meals["timestamp"] = START + pd.to_timedelta(meals.minute, unit="min")
    # phone-app meal log: some meals skipped, carbs estimated by eye, time rounded
    comp = dict(zip(cohort.patient_id, [l["log_compliance"] for l in cohort["_life"]]))
    logged = meals[rng.random(len(meals)) < meals.patient_id.map(comp)].copy()
    logged["carbs_logged_g"] = (np.round(logged.carbs_g * np.exp(rng.normal(0, 0.22, len(logged))) / 5) * 5).clip(5)
    logged["timestamp"] = (logged.timestamp + pd.to_timedelta(rng.normal(0, 6, len(logged)), unit="min")).dt.round("min")
    meal_log = logged[["patient_id", "timestamp", "slot", "dish", "carbs_logged_g", "gi"]].sort_values(["patient_id", "timestamp"])
    meals_true = meals[["patient_id", "timestamp", "slot", "dish", "carbs_g", "gi"]].sort_values(["patient_id", "timestamp"])
    days = pd.DataFrame(day_rows)
    days = days[days.day >= 1]
    return ts_df.reset_index(drop=True), meal_log.reset_index(drop=True), meals_true.reset_index(drop=True), days.reset_index(drop=True)


def add_labs(cohort, ts_df, seed=13):
    """EHR lab values that are consistent with each person's simulated glycaemia."""
    rng = np.random.default_rng(seed)
    g = ts_df.groupby("patient_id")["_true_glucose"]
    mean_g = g.mean()
    fasting = ts_df[ts_df.timestamp.dt.hour == 7].groupby("patient_id")["_true_glucose"].mean()
    c = cohort.copy()
    c["hba1c_pct"] = [round(float((mean_g[p] + 46.7) / 28.7 + rng.normal(0, 0.22)), 1) for p in c.patient_id]
    c["fpg_mgdl"] = [int(fasting[p] + rng.normal(0, 7)) for p in c.patient_id]
    c["hba1c_6m_ago"] = (c.hba1c_pct + rng.normal(0.15, 0.35, len(c))).round(1)
    c["hba1c_12m_ago"] = (c.hba1c_6m_ago + rng.normal(0.1, 0.35, len(c))).round(1)
    return c
