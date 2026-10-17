"""Step 4: figures for the README, presentation and video (single blue colour family).

    python -m glucosentinel.figures
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from .explain import friendly, GROUP_NAME
from .twin import PatientTwin, build_inputs
from .foods import FOODS, personal_portion
from .store import read_by_patient

ROOT = Path(__file__).resolve().parents[2]
FIG = ROOT / "reports/figures"
# one colour family, dark to light
C = ["#0B2545", "#13315C", "#1D4E89", "#2F6DB5", "#5B93D3", "#8FB8E8", "#C3DAF3", "#E6F0FB"]
EX_PID, EX_DAY = "PT-0054", "2026-09-10"

plt.rcParams.update({"font.family": ["Liberation Sans", "Arial", "DejaVu Sans"], "font.size": 10,
                     "axes.edgecolor": C[1], "axes.labelcolor": C[0], "xtick.color": C[1], "ytick.color": C[1],
                     "axes.spines.top": False, "axes.spines.right": False, "axes.titleweight": "bold",
                     "axes.titlecolor": C[0], "axes.titlesize": 11, "savefig.dpi": 200,
                     "savefig.bbox": "tight", "legend.frameon": False})


def save(fig, name):
    fig.savefig(FIG / f"{name}.png", facecolor="white")
    plt.close(fig)


def ablation(m, size=None):
    a = m["ablation_test"]
    order = ["CGM trend alarm (projected 30 min)", "Twin physiology alone", "EHR only", "Wearables only",
             "EHR + Wearables (fusion)", "GlucoSentinel: EHR + Wearables + Twin"]
    labels = ["CGM trend alarm", "Twin physiology alone", "EHR only", "Wearables only",
              "EHR + Wearables", "GlucoSentinel (EHR + Wearables + Twin)"]
    fig, axes = plt.subplots(1, 2, figsize=size or (10, 3.4), sharey=True)
    for ax, k, title in ((axes[0], "auroc", "AUROC"), (axes[1], "auprc", "AUPRC")):
        v = [a[o][k] for o in order]
        cols = [C[6], C[5], C[5], C[4], C[3], C[0]]
        b = ax.barh(labels, v, color=cols)
        ax.set_xlim(0.4 if k == "auroc" else 0.2, 1.0)
        ax.set_title(f"{title} on 24 unseen test patients")
        for r, x in zip(b, v):
            ax.text(x + 0.005, r.get_y() + r.get_height() / 2, f"{x:.3f}", va="center", fontsize=9, color=C[0])
    save(fig, "fig_ablation")


def alarms(m, size=None):
    s, f, t = m["episodes_test_glucosentinel"], m["episodes_test_fusion_only"], m["episodes_test_cgm_trend_alarm"]
    names = ["Trend\nalarm", "EHR +\nwearables", "Gluco\nSentinel"]
    fig, axes = plt.subplots(1, 3, figsize=size or (10.5, 3.0))
    for ax, key, title, fmt in ((axes[0], "warned_30min_ahead_pct", "Spikes warned\n30+ min ahead (%)", "{:.0f}%"),
                                (axes[1], "median_lead_min", "Median warning\nlead time (min)", "{:.0f}"),
                                (axes[2], "false_alerts_per_day", "False alerts\nper patient-day", "{:.2f}")):
        v = [t[key], f[key], s[key]]
        b = ax.bar(names, v, color=[C[6], C[4], C[0]], width=0.62)
        ax.set_title(title, fontsize=10)
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)
        for r, x in zip(b, v):
            ax.text(r.get_x() + r.get_width() / 2, x, fmt.format(x), ha="center", va="bottom", fontsize=10,
                    color=C[0], fontweight="bold")
        ax.tick_params(axis="x", labelsize=8)
    save(fig, "fig_alarm_comparison")


def forecast(m, size=None):
    fc = m["forecast_test"]
    hs = list(fc.keys())
    x = np.arange(len(hs)); w = 0.26
    fig, ax = plt.subplots(figsize=size or (6.4, 3.2))
    per = [fc[h]["persistence_rmse"] for h in hs]
    tw = [fc[h]["twin_rmse"] or np.nan for h in hs]
    sk = [fc[h]["glucosentinel_rmse"] for h in hs]
    ax.bar(x - w, per, w, color=C[6], label="Persistence (no change)")
    ax.bar(x, tw, w, color=C[4], label="Calibrated twin alone")
    ax.bar(x + w, sk, w, color=C[0], label="GlucoSentinel (twin + ML)")
    for xi, v in zip(x + w, sk):
        ax.text(xi, v + 0.8, f"{v:.0f}", ha="center", fontsize=8, color=C[0])
    ax.set_xticks(x, [f"+{h}" for h in hs]); ax.set_ylabel("RMSE (mg/dL)")
    ax.set_title("Glucose forecast error by horizon (test patients)")
    ax.legend(fontsize=8, loc="upper left")
    save(fig, "fig_forecast_rmse")


def clarke(m, size=None):
    d = pd.read_csv(ROOT / "reports/clarke_sample_60m.csv")
    fig, ax = plt.subplots(figsize=size or (4.4, 4.2))
    ax.scatter(d.reference, d.predicted, s=3, color=C[2], alpha=0.35, linewidths=0)
    lw = dict(color=C[0], lw=0.8)
    ax.plot([0, 400], [0, 400], ls=":", **lw)
    ax.plot([0, 175 / 3], [70, 70], **lw); ax.plot([175 / 3, 400 / 1.2], [70, 400], **lw)
    ax.plot([70, 70], [84, 400], **lw); ax.plot([70, 290], [180, 400], **lw); ax.plot([70, 70], [0, 56], **lw)
    ax.plot([70, 400], [56, 320], **lw); ax.plot([180, 180], [0, 70], **lw); ax.plot([180, 400], [70, 70], **lw)
    ax.plot([240, 240], [70, 180], **lw); ax.plot([240, 400], [180, 180], **lw); ax.plot([130, 180], [0, 70], **lw)
    for txt, xy in (("A", (35, 15)), ("B", (300, 380)), ("B", (370, 255)), ("C", (165, 380)), ("C", (160, 15)),
                    ("D", (30, 140)), ("D", (380, 120)), ("E", (30, 380)), ("E", (380, 15))):
        ax.text(*xy, txt, fontsize=12, fontweight="bold", color=C[1])
    ax.set_xlim(0, 400); ax.set_ylim(0, 400)
    ax.set_xlabel("Actual CGM in 60 min (mg/dL)"); ax.set_ylabel("Predicted (mg/dL)")
    z = m["clarke_grid_60min_pct"]
    ax.set_title(f"Clarke error grid, 60-min forecast\nZone A {z['A']}%, A+B {z['A'] + z['B']:.1f}%")
    save(fig, "fig_clarke")


def importance(m, size=None, bars_only=False):
    top = pd.DataFrame(m["top_features"])
    top["name"] = top.feature.map(friendly)
    top = top.groupby("name", as_index=False).mean_abs_shap.sum().sort_values("mean_abs_shap", ascending=False).head(14)[::-1]
    share = m["importance_share_by_group_pct"]
    if bars_only:
        fig, a1 = plt.subplots(figsize=size or (7, 4))
        a1.barh(top.name, top.mean_abs_shap, color=C[2])
        a1.set_title("Mean |SHAP| on test rows"); a1.tick_params(axis="y", labelsize=9)
        save(fig, "fig_importance_bars")
        return
    fig, (a1, a2) = plt.subplots(1, 2, figsize=size or (10.5, 4.0), gridspec_kw=dict(width_ratios=[2.2, 1]))
    a1.barh(top.name, top.mean_abs_shap, color=C[2])
    a1.set_title("What drives each prediction (mean |SHAP|)"); a1.tick_params(axis="y", labelsize=8.5)
    keys = ["R", "T", "W", "E"]
    vals = [share.get(k, 0) for k in keys]
    a2.pie(vals, labels=[f"{GROUP_NAME[k]}\n{v:.0f}%" for k, v in zip(keys, vals)], colors=[C[0], C[2], C[4], C[6]],
           startangle=90, wedgeprops=dict(width=0.42, edgecolor="white"), textprops=dict(fontsize=8, color=C[0]))
    a2.set_title("Share of signal by data stream")
    save(fig, "fig_importance")


def recovery(size=None):
    r = pd.read_csv(ROOT / "reports/twin_parameter_recovery.csv")
    fig, ax = plt.subplots(figsize=size or (4.4, 4.0))
    ax.scatter(r.true_SI * 1e4, r.prior_SI * 1e4, s=14, color=C[5], label="EHR prior only")
    ax.scatter(r.true_SI * 1e4, r.twin_SI * 1e4, s=14, color=C[0], label="Calibrated twin")
    lim = [0, max(r.true_SI.max(), r.twin_SI.max()) * 1e4 * 1.05]
    ax.plot(lim, lim, ls=":", color=C[1], lw=0.8)
    ax.set_xlabel("True insulin sensitivity (simulator)"); ax.set_ylabel("Twin estimate")
    ax.set_title("The twin learns each patient's physiology")
    ax.legend(fontsize=8)
    save(fig, "fig_parameter_recovery")


def _example_frames():
    ts = pd.read_parquet(ROOT / "data/wearables/cgm_wearables_5min.parquet")
    ts = ts[ts.patient_id == EX_PID].reset_index(drop=True)
    meals = pd.read_csv(ROOT / "data/wearables/meal_log.csv", parse_dates=["timestamp"])
    meals = meals[meals.patient_id == EX_PID]
    pred = read_by_patient(ROOT / "data/processed/predictions", EX_PID)
    return ts, meals, pred


def example_day(m, thr, size=None):
    ts, meals, pred = _example_frames()
    d0 = pd.Timestamp(EX_DAY); d1 = d0 + pd.Timedelta(days=1)
    t = ts[(ts.timestamp >= d0) & (ts.timestamp < d1)]
    p = pred[(pred.timestamp >= d0) & (pred.timestamp < d1)]
    mm = meals[(meals.timestamp >= d0) & (meals.timestamp < d1)]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=size or (10, 4.6), sharex=True, gridspec_kw=dict(height_ratios=[2.2, 1]))
    a1.axhspan(180, 400, color=C[7]); a1.axhline(180, color=C[3], lw=0.8, ls="--")
    a1.plot(t.timestamp, t.cgm_mgdl, color=C[0], lw=1.6, label="CGM glucose")
    a1.plot(p.timestamp + pd.Timedelta(minutes=60), p.fc_60m, color=C[4], lw=1.0, ls="-",
            label="GlucoSentinel forecast made 60 min earlier")
    for _, r in mm.iterrows():
        a1.annotate(r.dish.split(" (")[0].split(" with")[0], (r.timestamp, 75), fontsize=7, color=C[1], rotation=90,
                    va="bottom", ha="center")
        a1.axvline(r.timestamp, color=C[5], lw=0.6)
    a1.set_ylim(70, max(300, t.cgm_mgdl.max() + 10)); a1.set_ylabel("mg/dL")
    a1.set_title(f"One day for test patient {EX_PID}: the warning arrives well before the spike")
    a1.legend(fontsize=8, loc="upper left", ncol=2)
    pe = np.where(p.eligible, p.p_spike, np.nan)
    a2.fill_between(p.timestamp, 0, pe, color=C[4], alpha=0.6, step="mid")
    a2.axhline(thr, color=C[0], lw=0.9, ls="--"); a2.text(p.timestamp.iloc[5], thr + 0.03, "alert threshold", fontsize=7, color=C[0])
    al = p[(p.p_spike >= thr) & p.eligible]
    first = al[al.timestamp.diff().fillna(pd.Timedelta("1h")) > pd.Timedelta("60min")]
    a2.scatter(first.timestamp, [1.02] * len(first), marker="v", color=C[0], s=30, zorder=5, clip_on=False)
    a2.set_ylim(0, 1.05); a2.set_ylabel("Spike risk\nnext 2 h")
    a2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    save(fig, "fig_example_day")


def streams(size=None):
    ts, meals, _ = _example_frames()
    d0 = pd.Timestamp(EX_DAY) - pd.Timedelta(hours=6); d1 = d0 + pd.Timedelta(hours=30)
    t = ts[(ts.timestamp >= d0) & (ts.timestamp < d1)]
    fig, ax = plt.subplots(4, 1, figsize=size or (10, 5.2), sharex=True, gridspec_kw=dict(height_ratios=[1.6, 1, 1, 0.6]))
    ax[0].plot(t.timestamp, t.cgm_mgdl, color=C[0]); ax[0].set_ylabel("CGM\nmg/dL")
    ax[1].bar(t.timestamp, t.steps, width=0.0034, color=C[3]); ax[1].set_ylabel("Steps\n/5 min")
    ax[2].plot(t.timestamp, t.heart_rate_bpm, color=C[2], lw=0.8, label="Heart rate")
    a2b = ax[2].twinx(); a2b.plot(t.timestamp, t.hrv_rmssd_ms, color=C[5], lw=0.8); a2b.set_ylabel("HRV ms", color=C[4])
    a2b.spines["right"].set_visible(True)
    ax[2].set_ylabel("HR bpm")
    stage = t.sleep_stage.astype(str).map({"deep": 3, "light": 2, "rem": 1, "awake": 0, "out_of_bed": np.nan})
    ax[3].step(t.timestamp, stage, color=C[1], where="mid"); ax[3].set_yticks([0, 1, 2, 3], ["awake", "REM", "light", "deep"], fontsize=7)
    ax[3].set_ylabel("Sleep")
    ax[3].xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax[0].set_title("Dynamic stream: 5-minute CGM, steps, heart rate, HRV and sleep stages")
    save(fig, "fig_data_streams")


def whatif(size=None):
    tj = (ROOT / f"models/twins/{EX_PID}.json").read_text()
    twin = PatientTwin.from_json(tj)
    ts, meals, _ = _example_frames()
    inp = build_inputs(ts, meals, ts.timestamp.iloc[0], len(ts))
    st = twin.filter(inp)
    now = pd.Timestamp(EX_DAY) + pd.Timedelta(hours=12, minutes=30)
    k = int((now - ts.timestamp.iloc[0]) / pd.Timedelta("5min"))
    lunch = "Rice with sambar and poriyal"; swap = "Bajra or jowar roti with dal"
    L, S = FOODS[lunch], FOODS[swap]
    pf = personal_portion(meals)
    sc = [dict(meals=[(30, L["carbs"] * pf, L["gi"])]),
          dict(meals=[(30, L["carbs"] * pf, L["gi"])], walks=[(50, 20, 100)]),
          dict(meals=[(30, S["carbs"] * pf, S["gi"])]),
          dict(meals=[(30, S["carbs"] * pf, S["gi"])], walks=[(50, 20, 100)])]
    names = ["Usual: rice, sambar, poriyal", "Same lunch + 20-min walk", "Swap to bajra roti + dal", "Swap + 20-min walk"]
    out = twin.what_if(st[:, k], now.hour + now.minute / 60, sc, hours_ahead=4)
    x = np.arange(out.shape[1]) * 5 + 5
    fig, ax = plt.subplots(figsize=size or (7.2, 3.4))
    ax.axhspan(180, 400, color=C[7]); ax.axhline(180, color=C[3], lw=0.8, ls="--")
    for y, n, c, lw in zip(out, names, [C[0], C[2], C[4], C[5]], [2.2, 1.6, 1.6, 1.6]):
        ax.plot(x, y, color=c, lw=lw, label=f"{n}: peak {y.max():.0f}")
    ax.set_ylim(min(100, out.min() - 10), out.max() + 115)
    ax.set_xlabel("Minutes from now (12:30, lunch at 13:00)"); ax.set_ylabel("mg/dL")
    ax.set_title(f"What-if simulator on {EX_PID}'s digital twin")
    ax.legend(fontsize=7.5, loc="upper right", bbox_to_anchor=(1.0, 1.0))
    save(fig, "fig_whatif")
    return {n: dict(peak=round(float(y.max())), minutes_above_180=int((y > 180).sum() * 5)) for n, y in zip(names, out)}


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    m = json.loads((ROOT / "reports/metrics.json").read_text())
    thr = json.loads((ROOT / "models/feature_list.json").read_text())["threshold"]
    ablation(m); alarms(m); forecast(m); clarke(m); importance(m); importance(m, bars_only=True); recovery()
    example_day(m, thr); streams()
    w = whatif()
    m["whatif_example"] = dict(patient=EX_PID, time=f"{EX_DAY} 12:30", scenarios=w)
    (ROOT / "reports/metrics.json").write_text(json.dumps(m, indent=1))
    print("figures written:", sorted(p.name for p in FIG.glob("*.png")))
    print(w)


if __name__ == "__main__":
    main()
