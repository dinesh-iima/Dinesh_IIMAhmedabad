"""Step 3: train the 2-hour spike predictor and glucose forecasters, run the ablation study
(EHR only vs wearables only vs fused vs fused + twin), and write metrics and models.

    python -m glucosentinel.train

Evaluation protocol
  * Patient-level split: 84 train / 12 validation / 24 test patients (no person appears in two sets).
  * Week 1 of every patient is the onboarding week (twin calibration, routine learning).
  * All models are trained and scored only on week 2, so nothing the model sees was used
    to fit the twin or the routine features.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss, precision_recall_curve

from .features import feature_groups, spike_episodes, H
from .store import read_by_patient, write_by_patient

ROOT = Path(__file__).resolve().parents[2]
SEED = 42
PARAMS = dict(objective="binary", learning_rate=0.04, num_leaves=31, min_data_in_leaf=80,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              verbose=-1, seed=SEED, num_threads=2)


def split_patients(pids):
    rng = np.random.default_rng(SEED)
    p = np.array(sorted(pids)); rng.shuffle(p)
    return set(p[:84]), set(p[84:96]), set(p[96:])


def fit_clf(tr, va, cols, rounds=1500):
    dtr = lgb.Dataset(tr[cols], tr.y_spike_2h)
    dva = lgb.Dataset(va[cols], va.y_spike_2h, reference=dtr)
    return lgb.train(PARAMS, dtr, rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(80, verbose=False)])


def scores(y, p):
    return dict(auroc=round(float(roc_auc_score(y, p)), 4), auprc=round(float(average_precision_score(y, p)), 4))


def episode_metrics(df, prob_col, thr):
    """Clinically meaningful view: how many spikes were warned about, how early, at what alarm cost."""
    det, leads, n_ep, true_alerts, false_alerts, days = 0, [], 0, 0, 0, 0.0
    for pid, g in df.groupby("patient_id"):
        g = g.reset_index(drop=True)
        onsets, _ = spike_episodes(g.cgm_raw.to_numpy())
        p = np.where(g.eligible, g[prob_col], 0.0)
        alert = p >= thr
        for o in onsets:
            w0 = max(o - H, 0)
            if not g.eligible.iloc[w0:o].any():
                continue
            n_ep += 1
            hits = np.where(alert[w0:o])[0]
            if len(hits):
                lead = (o - (w0 + hits[0])) * 5
                leads.append(lead)
                det += lead >= 30
        last = -10_000
        for i in np.where(alert)[0]:
            rising = i == 0 or not alert[i - 1]
            if not rising or i - last <= 12:       # one notification per hour at most
                continue
            last = i
            if ((onsets > i) & (onsets <= i + H)).any():
                true_alerts += 1
            else:
                false_alerts += 1
        days += len(g) / 288
    return dict(spike_episodes=n_ep,
                warned_30min_ahead_pct=round(100 * det / max(n_ep, 1), 1),
                warned_any_lead_pct=round(100 * len(leads) / max(n_ep, 1), 1),
                median_lead_min=float(np.median(leads)) if leads else 0.0,
                alert_precision_pct=round(100 * true_alerts / max(true_alerts + false_alerts, 1), 1),
                false_alerts_per_day=round(false_alerts / max(days, 1), 2),
                alerts_per_day=round((true_alerts + false_alerts) / max(days, 1), 2))


def clarke_zone(ref, pred):
    """Clarke error grid zone (A to E) for each (reference, predicted) pair in mg/dL."""
    z = []
    for r, p in zip(ref, pred):
        if (r <= 70 and p <= 70) or (0.8 * r <= p <= 1.2 * r):
            z.append("A")
        elif (r >= 180 and p <= 70) or (r <= 70 and p >= 180):
            z.append("E")
        elif ((70 <= r <= 290) and p >= r + 110) or ((130 <= r <= 180) and p <= (7 / 5) * r - 182):
            z.append("C")
        elif (r >= 240 and 70 <= p <= 180) or (r <= 175 / 3 and 70 <= p <= 180) or \
                ((175 / 3 <= r <= 70) and p >= (6 / 5) * r):
            z.append("D")
        else:
            z.append("B")
    return np.array(z)


def main():
    feats = read_by_patient(ROOT / "data/processed/features")
    feats["cgm_raw"] = feats.groupby("patient_id")["target_cgm_30m"].shift(6)  # raw CGM at t (gaps kept)
    groups = feature_groups(feats.columns)
    tr_p, va_p, te_p = split_patients(feats.patient_id.unique())
    wk2 = feats[feats.eval_week]
    elig = wk2[wk2.eligible]
    tr, va, te = (elig[elig.patient_id.isin(s)] for s in (tr_p, va_p, te_p))
    time_cols = ["hour_sin", "hour_cos", "weekend"]
    configs = {
        "EHR only": groups["E"] + time_cols,
        "Wearables only": groups["W"],
        "EHR + Wearables (fusion)": groups["E"] + groups["W"],
        "GlucoSentinel: EHR + Wearables + Twin": groups["E"] + groups["W"] + groups["R"] + groups["T"],
    }
    results, models = {}, {}
    for name, cols in configs.items():
        m = fit_clf(tr, va, cols)
        models[name] = (m, cols)
        results[name] = scores(te.y_spike_2h, m.predict(te[cols], num_iteration=m.best_iteration))
        results[name]["n_features"] = len(cols)
        print(f"{name:34s} {results[name]}")
    # non-ML reference points
    roc_proj = te.cgm_now + 30 * te.cgm_slope_30m
    results["CGM trend alarm (projected 30 min)"] = scores(te.y_spike_2h, roc_proj.fillna(te.cgm_now))
    results["Twin physiology alone"] = scores(te.y_spike_2h, te.twin_habit_max_2h)

    full_name = "GlucoSentinel: EHR + Wearables + Twin"
    m, cols = models[full_name]
    m.save_model(str(ROOT / "models/spike_model.txt"), num_iteration=m.best_iteration)

    # predictions for every week-2 row of every patient (dashboard), flagged by split
    allp = wk2.copy()
    allp["p_spike"] = m.predict(allp[cols], num_iteration=m.best_iteration)
    mf, cf = models["EHR + Wearables (fusion)"]
    allp["p_fusion"] = mf.predict(allp[cf], num_iteration=mf.best_iteration)
    allp["p_trend"] = (allp.cgm_now + 30 * allp.cgm_slope_30m).fillna(allp.cgm_now)
    allp["split"] = np.where(allp.patient_id.isin(te_p), "test", np.where(allp.patient_id.isin(va_p), "validation", "train"))
    val_rows, test_rows = allp[allp.split == "validation"], allp[allp.split == "test"]

    def pick_threshold(col, max_false_per_day=1.0):
        """Operating point chosen on validation patients only: catch as many spikes as possible
        while sending at most about one false alert per patient per day."""
        best = None
        for t in np.round(np.arange(0.20, 0.96, 0.025), 3):
            e = episode_metrics(val_rows, col, t)
            if e["false_alerts_per_day"] <= max_false_per_day:
                if best is None or e["warned_30min_ahead_pct"] > best[1]:
                    best = (float(t), e["warned_30min_ahead_pct"])
        return best[0] if best else 0.95

    thr = pick_threshold("p_spike")
    thr_f = pick_threshold("p_fusion")
    ep = episode_metrics(test_rows, "p_spike", thr)
    ep_fusion = episode_metrics(test_rows, "p_fusion", thr_f)
    ep_trend = episode_metrics(test_rows, "p_trend", 180.0)
    print("episodes (GlucoSentinel):", thr, ep)
    print("episodes (fusion):", thr_f, ep_fusion)
    print("episodes (CGM trend alarm):", ep_trend)

    # glucose forecasters (delta from now) for the dashboard curve
    fc_res = {}
    rows_all = wk2[wk2.cgm_now.notna()]
    for hmin in (30, 60, 90, 120):
        tgt = f"target_cgm_{hmin}m"
        d = rows_all[rows_all[tgt].notna()]
        dtr, dva, dte = (d[d.patient_id.isin(s)] for s in (tr_p, va_p, te_p))
        rp = dict(PARAMS, objective="regression_l1")
        r = lgb.train(rp, lgb.Dataset(dtr[cols], dtr[tgt] - dtr.cgm_now), 1500,
                      valid_sets=[lgb.Dataset(dva[cols], dva[tgt] - dva.cgm_now)],
                      callbacks=[lgb.early_stopping(80, verbose=False)])
        r.save_model(str(ROOT / f"models/forecast_{hmin}m.txt"), num_iteration=r.best_iteration)
        pred = dte.cgm_now + r.predict(dte[cols], num_iteration=r.best_iteration)
        allp[f"fc_{hmin}m"] = allp.cgm_now + r.predict(allp[cols], num_iteration=r.best_iteration)
        twin_col = {30: "twin_fc_30m", 60: "twin_fc_60m", 120: "twin_fc_120m"}.get(hmin)
        rm = lambda a: round(float(np.sqrt(np.mean((a - dte[tgt]) ** 2))), 1)
        fc_res[f"{hmin} min"] = dict(persistence_rmse=rm(dte.cgm_now),
                                     twin_rmse=rm(dte[twin_col]) if twin_col else None,
                                     glucosentinel_rmse=rm(pred),
                                     glucosentinel_mae=round(float(np.mean(np.abs(pred - dte[tgt]))), 1))
        if hmin == 60:
            z = clarke_zone(dte[tgt].to_numpy(), pred.to_numpy())
            clarke = {k: round(100 * float(np.mean(z == k)), 1) for k in "ABCDE"}
            pd.DataFrame({"reference": dte[tgt].to_numpy(), "predicted": pred.to_numpy()}) \
                .sample(4000, random_state=1).to_csv(ROOT / "reports/clarke_sample_60m.csv", index=False)
    print("forecast:", fc_res, "clarke60:", clarke)

    # explainability: mean |SHAP| on test rows
    sample = te.sample(min(20000, len(te)), random_state=1)
    contrib = m.predict(sample[cols], num_iteration=m.best_iteration, pred_contrib=True)[:, :-1]
    imp = pd.Series(np.abs(contrib).mean(0), index=cols).sort_values(ascending=False)
    gmap = {c: g for g, cs in groups.items() for c in cs}
    group_share = imp.groupby(imp.index.map(lambda c: gmap.get(c, "W"))).sum()
    group_share = (100 * group_share / group_share.sum()).round(1).to_dict()

    # subgroup fairness check (test patients)
    ehr = pd.read_csv(ROOT / "data/ehr/patients.csv")
    tt = test_rows[test_rows.eligible].merge(ehr[["patient_id", "sex", "region", "diagnosis"]], on="patient_id")
    sub = {}
    for col in ("sex", "region", "diagnosis"):
        for v, g in tt.groupby(col):
            if g.y_spike_2h.nunique() == 2:
                sub[f"{col}={v}"] = dict(n_patients=int(g.patient_id.nunique()),
                                         auroc=round(float(roc_auc_score(g.y_spike_2h, g.p_spike)), 3))

    # twin parameter recovery vs simulator ground truth (all 120 patients)
    hidden = json.loads((ROOT / "data/ground_truth/hidden_physiology.json").read_text())
    rec = []
    for pid, hp in hidden.items():
        tj = json.loads((ROOT / f"models/twins/{pid}.json").read_text())
        rec.append(dict(patient_id=pid, true_SI=hp["params"]["SI"], prior_SI=tj["prior"]["SI"], twin_SI=tj["params"]["SI"],
                        true_gamma=hp["params"]["gamma"], prior_gamma=tj["prior"]["gamma"], twin_gamma=tj["params"]["gamma"],
                        fit_rmse=tj["fit"]["forecast_rmse_mgdl"]))
    rec = pd.DataFrame(rec)
    rec.to_csv(ROOT / "reports/twin_parameter_recovery.csv", index=False)
    corr = lambda a, b: round(float(np.corrcoef(np.log(rec[a]), np.log(rec[b]))[0, 1]), 3)
    recovery = dict(insulin_sensitivity=dict(ehr_prior_r=corr("prior_SI", "true_SI"), calibrated_twin_r=corr("twin_SI", "true_SI")),
                    beta_cell_response=dict(ehr_prior_r=corr("prior_gamma", "true_gamma"), calibrated_twin_r=corr("twin_gamma", "true_gamma")))

    metrics = dict(
        task="Predict onset of a CGM glucose excursion above 180 mg/dL (lasting 15 min or more) within the next 120 minutes",
        cohort=dict(patients=int(feats.patient_id.nunique()), train=len(tr_p), validation=len(va_p), test=len(te_p),
                    days_per_patient=14, readings_5min=int(len(feats)),
                    test_rows=int(len(te)), test_positive_rate=round(float(te.y_spike_2h.mean()), 3)),
        ablation_test=results,
        operating_threshold=round(thr, 4),
        episodes_test_glucosentinel=ep,
        episodes_test_fusion_only=ep_fusion,
        episodes_test_cgm_trend_alarm=ep_trend,
        fusion_threshold=round(thr_f, 4),
        forecast_test=fc_res,
        clarke_grid_60min_pct=clarke,
        importance_share_by_group_pct=group_share,
        top_features=[{"feature": k, "mean_abs_shap": round(float(v), 4)} for k, v in imp.head(20).items()],
        subgroups_test=sub,
        twin_parameter_recovery=recovery,
    )
    (ROOT / "reports/metrics.json").write_text(json.dumps(metrics, indent=1))
    (ROOT / "models/feature_list.json").write_text(json.dumps(dict(features=cols, threshold=thr, groups=groups,
                                                                  split=dict(train=sorted(tr_p), validation=sorted(va_p), test=sorted(te_p))), indent=1))
    keep = ["patient_id", "timestamp", "split", "cgm_now", "cgm_raw", "eligible", "y_spike_2h", "p_spike", "p_fusion",
            "fc_30m", "fc_60m", "fc_90m", "fc_120m", "twin_fc_30m", "twin_fc_60m", "twin_fc_120m"]
    write_by_patient(allp[keep], ROOT / "data/processed/predictions")
    print(json.dumps({k: metrics[k] for k in ("importance_share_by_group_pct", "twin_parameter_recovery", "subgroups_test")}, indent=1))


if __name__ == "__main__":
    main()
