"""GlucoSentinel doctor console: how a clinician interacts with a patient's digital twin.

    streamlit run dashboard/app.py
"""
import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from glucosentinel.twin import PatientTwin, build_inputs, DT   # noqa: E402
from glucosentinel.foods import FOODS, SWAPS, personal_portion   # noqa: E402
from glucosentinel.explain import friendly   # noqa: E402
from glucosentinel.store import read_by_patient   # noqa: E402

C = ["#0B2545", "#13315C", "#1D4E89", "#2F6DB5", "#5B93D3", "#8FB8E8", "#C3DAF3", "#E6F0FB"]
st.set_page_config(page_title="GlucoSentinel Doctor Console", page_icon="🩺", layout="wide")
st.markdown(f"""
<style>
 .block-container {{padding-top: 3.2rem; padding-bottom: 1rem;}}
 html, body, [class*="css"] {{font-family: Arial, 'Liberation Sans', sans-serif;}}
 .card {{background:{C[7]}; border:1px solid {C[6]}; border-radius:10px; padding:10px 14px; height:100%;}}
 .card h4 {{margin:0 0 4px 0; color:{C[0]}; font-size:0.85rem; text-transform:uppercase; letter-spacing:0.04em;}}
 .big {{font-size:2.1rem; font-weight:700; color:{C[0]}; line-height:1.1;}}
 .sub {{color:{C[2]}; font-size:0.85rem;}}
 .chip {{display:inline-block; background:white; border:1px solid {C[5]}; color:{C[1]}; border-radius:12px;
        padding:1px 9px; margin:2px 3px 2px 0; font-size:0.78rem;}}
 .alert-high {{background:{C[0]}; color:white; border-radius:10px; padding:10px 14px;}}
 .alert-watch {{background:{C[3]}; color:white; border-radius:10px; padding:10px 14px;}}
 .alert-low {{background:{C[6]}; color:{C[0]}; border-radius:10px; padding:10px 14px;}}
 .alert-high .big, .alert-watch .big {{color:white;}}
 .alert-high .sub, .alert-watch .sub {{color:{C[7]};}}
</style>""", unsafe_allow_html=True)


# ------------------------------------------------------------------ data
@st.cache_data
def load_static():
    ehr = pd.read_csv(ROOT / "data/ehr/patients.csv")
    meta = json.loads((ROOT / "models/feature_list.json").read_text())
    metrics = json.loads((ROOT / "reports/metrics.json").read_text())
    pred = read_by_patient(ROOT / "data/processed/predictions")
    twins = {p.stem: json.loads(p.read_text()) for p in (ROOT / "models/twins").glob("*.json")}
    return ehr, meta, metrics, pred, twins


@st.cache_data
def load_patient(pid):
    ts = pd.read_parquet(ROOT / "data/wearables/cgm_wearables_5min.parquet", filters=[("patient_id", "==", pid)])
    meals = pd.read_csv(ROOT / "data/wearables/meal_log.csv", parse_dates=["timestamp"])
    meals = meals[meals.patient_id == pid]
    feats = read_by_patient(ROOT / "data/processed/features", pid)
    return ts, meals, feats


@st.cache_resource
def load_model():
    return lgb.Booster(model_file=str(ROOT / "models/spike_model.txt"))


@st.cache_data
def twin_states(pid, twin_json):
    ts, meals, _ = load_patient(pid)
    twin = PatientTwin.from_json(twin_json)
    inp = build_inputs(ts, meals, ts.timestamp.iloc[0], len(ts))
    return twin.filter(inp), inp["idx"]


ehr_all, META, METRICS, PRED, TWINS = load_static()
THR = META["threshold"]
COLS = META["features"]
split_of = {p: s for s, ps in META["split"].items() for p in ps}

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.markdown(f"### 🩺 GlucoSentinel\n<span class='sub'>Doctor console for the type 2 diabetes digital twin</span>",
                unsafe_allow_html=True)
    show_all = st.toggle("Include training patients", value=False)
    pool = ehr_all if show_all else ehr_all[ehr_all.patient_id.isin(META["split"]["test"])]
    lab = {r.patient_id: f"{r.patient_id} · {r.display_name}, {r.age}{r.sex} · HbA1c {r.hba1c_pct}%"
           for r in pool.itertuples()}
    default = list(lab).index("PT-0054") if "PT-0054" in lab else 0
    pid = st.selectbox("Patient", list(lab), index=default, format_func=lab.get)
    days = sorted(PRED[PRED.patient_id == pid].timestamp.dt.date.unique())
    day = st.selectbox("Day (monitoring week)", days, index=min(2, len(days) - 1),
                       format_func=lambda d: pd.Timestamp(d).strftime("%a %d %b %Y"))
    tod = st.slider("Time now", 0.0, 23.0 + 55 / 60, 12.5, step=5 / 60,
                    format="%.2f h", help="Replay the day: move 'now' to see what the doctor sees at that moment")
    reveal = st.toggle("Validation mode: reveal what actually happened", value=True)
    st.caption(f"Patient set: **{split_of.get(pid, 'test')}** · all data synthetic · alert threshold {THR:.2f}")

now = pd.Timestamp(day) + pd.Timedelta(minutes=int(round(tod * 60 / 5) * 5))
ts, meals, feats = load_patient(pid)
P = ehr_all.set_index("patient_id").loc[pid]
tj = TWINS[pid]
twin = PatientTwin.from_json(json.dumps(tj))
states, idx = twin_states(pid, json.dumps(tj))
k_now = int(np.clip(idx.get_indexer([now], method="nearest")[0], 0, len(idx) - 1))
row = feats[feats.timestamp == now]
pr = PRED[(PRED.patient_id == pid)].set_index("timestamp")
prow = pr.loc[now] if now in pr.index else None

# ------------------------------------------------------------------ header
c1, c2, c3 = st.columns([1.35, 1.25, 1.1])
with c1:
    meds = [m for m, f in (("Metformin", P.on_metformin), ("Glimepiride", P.on_sulfonylurea), ("Sitagliptin", P.on_dpp4i)) if f]
    chips = "".join(f"<span class='chip'>{x}</span>" for x in
                    [P.diagnosis + (f", {P.diabetes_duration_y} y" if P.diabetes_duration_y else ""),
                     f"BMI {P.bmi}", f"HbA1c {P.hba1c_pct}%", f"FPG {P.fpg_mgdl}", f"BP {P.sbp}/{P.dbp}",
                     f"TCF7L2 risk alleles {P.tcf7l2_rs7903146_risk_alleles}", f"PRS {P.t2d_prs:+.2f}"] +
                    meds + (["Hypertension"] if P.hypertension else []) + (["Family history"] if P.family_history_t2d else []))
    st.markdown(f"<div class='card'><h4>Patient record (EHR / FHIR)</h4><div class='big' style='font-size:1.4rem'>"
                f"{P.display_name} <span class='sub'>{P.age} y, {'male' if P.sex == 'M' else 'female'}, {P.city}</span></div>"
                f"{chips}</div>", unsafe_allow_html=True)
with c2:
    allsi = np.array([t["params"]["SI"] for t in TWINS.values()])
    allg = np.array([t["params"]["gamma"] for t in TWINS.values()])
    pct = lambda a, v: int(round(100 * (a < v).mean()))
    p_ = tj["params"]
    st.markdown(
        f"<div class='card'><h4>Virtual patient (calibrated twin)</h4>"
        f"<span class='chip'>Insulin sensitivity: {pct(allsi, p_['SI'])}th pct</span>"
        f"<span class='chip'>Beta-cell response: {pct(allg, p_['gamma'])}th pct</span>"
        f"<span class='chip'>Basal glucose {p_['Gb']:.0f} mg/dL</span>"
        f"<span class='chip'>Dawn effect +{p_['dawn']:.0f} mg/dL</span>"
        f"<span class='chip'>Usual portion x{personal_portion(meals):.2f}</span>"
        f"<div class='sub' style='margin-top:6px'>Calibrated on a 7-day onboarding week. "
        f"Twin fit error {tj['fit']['forecast_rmse_mgdl']} mg/dL. Sensitivity is "
        f"{p_['SI'] / tj['prior']['SI']:.2f}x what the EHR alone suggested.</div></div>", unsafe_allow_html=True)
with c3:
    if prow is not None and bool(prow.eligible):
        risk = float(prow.p_spike)
        fc = [prow.fc_30m, prow.fc_60m, prow.fc_90m, prow.fc_120m]
        first = next((h for h, v in zip((30, 60, 90, 120), fc) if v > 180), None)
        cls, label = ("alert-high", "HIGH: alert sent") if risk >= THR else (("alert-watch", "WATCH") if risk >= 0.4 else ("alert-low", "LOW"))
        msg = f"Spike above 180 expected in about {first} min" if first else f"Forecast peak {max(fc):.0f} mg/dL in next 2 h"
        st.markdown(f"<div class='{cls}'><h4 style='color:inherit'>Spike risk, next 2 hours</h4>"
                    f"<div class='big'>{risk * 100:.0f}% <span style='font-size:1rem'>{label}</span></div>"
                    f"<div class='sub'>Now {prow.cgm_now:.0f} mg/dL · {msg}</div></div>", unsafe_allow_html=True)
    else:
        g = ts.set_index("timestamp").cgm_mgdl.get(now, np.nan)
        st.markdown(f"<div class='alert-high'><h4 style='color:inherit'>In a spike now</h4><div class='big'>"
                    f"{g:.0f} mg/dL</div><div class='sub'>Prediction pauses while glucose is above 180</div></div>"
                    if g == g and g > 180 else
                    "<div class='alert-low'><h4>Spike risk</h4><div class='sub'>No prediction at this moment "
                    "(sensor gap or onboarding week)</div></div>", unsafe_allow_html=True)

# ------------------------------------------------------------------ main chart
st.write("")
left, right = st.columns([2.3, 1])
with left:
    w0, w1 = now - pd.Timedelta(hours=12), now + pd.Timedelta(hours=2.5)
    t = ts[(ts.timestamp >= w0) & (ts.timestamp <= (w1 if reveal else now))]
    past, fut = t[t.timestamp <= now], t[t.timestamp > now]
    fc_tw = twin.forecast(states[:, k_now:k_now + 1],
                          {"hours": np.array([now.hour + now.minute / 60]), "si_mult": np.ones(1)})[0]
    fig = make_subplots(rows=3, cols=1, shared_xaxes=True, row_heights=[0.62, 0.18, 0.2], vertical_spacing=0.04)
    fig.add_hrect(y0=180, y1=420, fillcolor=C[7], line_width=0, layer="below", row=1, col=1)
    fig.add_hline(y=180, line=dict(color=C[3], dash="dash", width=1), row=1, col=1)
    fig.add_trace(go.Scatter(x=past.timestamp, y=past.cgm_mgdl, name="CGM (past)", line=dict(color=C[0], width=2.4)), 1, 1)
    if reveal and len(fut):
        fig.add_trace(go.Scatter(x=fut.timestamp, y=fut.cgm_mgdl, name="What actually happened",
                                 line=dict(color=C[5], width=2, dash="dot")), 1, 1)
    fx = [now + pd.Timedelta(minutes=5 * (i + 1)) for i in range(len(fc_tw))]
    fig.add_trace(go.Scatter(x=fx, y=fc_tw, name="Twin physiology (no new food)", line=dict(color=C[4], width=1.6)), 1, 1)
    if prow is not None:
        fig.add_trace(go.Scatter(x=[now] + [now + pd.Timedelta(minutes=m) for m in (30, 60, 90, 120)],
                                 y=[prow.cgm_now, prow.fc_30m, prow.fc_60m, prow.fc_90m, prow.fc_120m],
                                 name="GlucoSentinel forecast", mode="lines+markers",
                                 line=dict(color=C[2], width=2.4), marker=dict(size=7)), 1, 1)
    mm = meals[(meals.timestamp >= w0) & (meals.timestamp <= now)]
    for r in mm.itertuples():
        fig.add_annotation(x=r.timestamp, y=95, text=f"🍽 {r.dish.split(' (')[0].split(' with')[0]} · {r.carbs_logged_g:.0f} g",
                           showarrow=False, textangle=-90, font=dict(size=9, color=C[1]), yanchor="bottom", row=1, col=1)
    fig.add_vline(x=now, line=dict(color=C[0], width=1))
    st_ = ts[(ts.timestamp >= w0) & (ts.timestamp <= now)]
    fig.add_trace(go.Bar(x=st_.timestamp, y=st_.steps, name="Steps / 5 min", marker_color=C[3], showlegend=False), 2, 1)
    rr = pr[(pr.index >= w0) & (pr.index <= now)]
    fig.add_trace(go.Scatter(x=rr.index, y=np.where(rr.eligible, rr.p_spike, np.nan), name="Spike risk",
                             fill="tozeroy", line=dict(color=C[4], width=1), showlegend=False), 3, 1)
    fig.add_hline(y=THR, line=dict(color=C[0], dash="dash", width=1), row=3, col=1)
    fig.update_yaxes(title_text="mg/dL", range=[70, max(300, float(np.nanmax(t.cgm_mgdl)) + 20)], row=1, col=1)
    fig.update_yaxes(title_text="steps", row=2, col=1)
    fig.update_yaxes(title_text="risk", range=[0, 1], row=3, col=1)
    fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white",
                      legend=dict(orientation="h", y=1.04, x=0, font=dict(size=11)), font=dict(family="Arial"))
    fig.update_xaxes(range=[w0, w1], gridcolor=C[7])
    st.plotly_chart(fig, width="stretch")

with right:
    st.markdown("<h4 style='margin:0;color:#0B2545;font-size:0.85rem;text-transform:uppercase'>Why the model thinks so</h4>", unsafe_allow_html=True)
    if len(row) and prow is not None and bool(prow.eligible):
        contrib = load_model().predict(row[COLS], pred_contrib=True)[0][:-1]
        s = pd.Series(contrib, index=[friendly(c) for c in COLS]).groupby(level=0).sum()
        s = s.reindex(s.abs().sort_values(ascending=False).index).head(7)[::-1]
        vals = {friendly(c): row[c].iloc[0] for c in COLS}
        f2 = go.Figure(go.Bar(x=s.values, y=[f"{n} ({vals.get(n, 0):.3g})" if not n.startswith("Time") else n for n in s.index],
                              orientation="h", marker_color=[C[0] if v > 0 else C[5] for v in s.values]))
        f2.update_layout(height=290, margin=dict(l=5, r=5, t=5, b=5), plot_bgcolor="white", font=dict(size=11, family="Arial"),
                         xaxis_title="dark raises risk, light lowers it", xaxis_title_font_size=10)
        st.plotly_chart(f2, width="stretch")
    else:
        st.caption("Drivers appear when a prediction is active.")
    lastn = feats[feats.timestamp == now]
    if len(lastn):
        r0 = lastn.iloc[0]
        st.markdown(
            f"<div class='card' style='margin-top:8px'><h4>Context from wearables</h4>"
            f"<span class='chip'>Sleep last night {r0.sleep_hours_last_night:.1f} h</span>"
            f"<span class='chip'>Awake in bed {r0.awake_in_bed_min_last_night:.0f} min</span>"
            f"<span class='chip'>Steps today {r0.steps_today:,.0f}</span>"
            f"<span class='chip'>HR {r0.hr_15m:.0f} bpm</span>"
            f"<span class='chip'>HRV {r0.hrv_rel_baseline * 100 if r0.hrv_rel_baseline == r0.hrv_rel_baseline else 100:.0f}% of baseline</span>"
            f"<span class='chip'>Usual carbs next 2 h: {r0.habit_carbs_next_2h:.0f} g</span>"
            f"<span class='chip'>Twin peak if usual meal: {r0.twin_habit_max_2h:.0f}</span></div>",
            unsafe_allow_html=True)

# ------------------------------------------------------------------ tabs
tab1, tab2, tab3, tab4 = st.tabs(["🧪 What-if simulator", "📈 Week review (AGP)", "⌚ Wearables", "📄 EHR record"])

with tab1:
    st.markdown("Run the patient's own twin forward from **now** under different plans. "
                "Use it to choose the advice that goes to the patient.")
    usual_slot = "lunch" if 10 <= tod < 15 else ("snack" if 15 <= tod < 18.5 else ("dinner" if tod >= 18.5 else "breakfast"))
    hist = meals[meals.slot == usual_slot].dish.value_counts()
    usual = hist.index[0] if len(hist) else next(d for d, f in FOODS.items() if f["slot"] == usual_slot)
    pf = personal_portion(meals)
    a, b = st.columns([1, 1.8])
    with a:
        dish = st.selectbox("Planned meal", list(FOODS), index=list(FOODS).index(usual))
        mins = st.slider("Eaten in (minutes from now)", 0, 120, 30, 5)
        portion = st.slider("Portion vs usual", 0.5, 1.5, 1.0, 0.05)
        walk = st.slider("Walk after the meal (minutes)", 0, 45, 0, 5)
        sleep = st.radio("Sleep tonight before tomorrow", ["As measured", "Poor night", "Good night"], horizontal=True,
                         help="Applies the population sleep effect on insulin sensitivity")
        med = st.selectbox("Medication what-if (illustrative)", ["No change", "Metformin up-titration", "Add DPP-4 inhibitor"])
    sif = {"As measured": 1.0, "Poor night": 0.78, "Good night": 1.08}[sleep]
    gbf = 1.0
    if med == "Metformin up-titration":
        sif *= 1.08; gbf = 0.94
    elif med == "Add DPP-4 inhibitor":
        gbf = 0.97
    U, D = FOODS[usual], FOODS[dish]
    swap = SWAPS.get(usual, usual); S = FOODS[swap]
    scen = [dict(meals=[(mins, U["carbs"] * pf, U["gi"])]),
            dict(meals=[(mins, D["carbs"] * pf * portion, D["gi"])],
                 walks=[(mins + 20, walk, 100)] if walk else [], si_factor=sif, gb_factor=gbf),
            dict(meals=[(mins, S["carbs"] * pf, S["gi"])], walks=[(mins + 20, 20, 100)])]
    names = [f"As usual: {usual}", "Doctor's plan (left)", f"Suggested: {swap} + 20-min walk"]
    out = twin.what_if(states[:, k_now], now.hour + now.minute / 60, scen, hours_ahead=4)
    with b:
        x = [now + pd.Timedelta(minutes=5 * (i + 1)) for i in range(out.shape[1])]
        f3 = go.Figure()
        f3.add_hrect(y0=180, y1=450, fillcolor=C[7], line_width=0, layer="below")
        f3.add_hline(y=180, line=dict(color=C[3], dash="dash", width=1))
        for y, n, c, w in zip(out, names, [C[0], C[3], C[5]], [3, 2.4, 2.4]):
            f3.add_trace(go.Scatter(x=x, y=y, name=f"{n} · peak {y.max():.0f}, {int((y > 180).sum() * DT)} min > 180",
                                    line=dict(color=c, width=w)))
        f3.update_layout(height=360, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor="white", font=dict(family="Arial"),
                         legend=dict(orientation="h", y=-0.18, font=dict(size=11)), yaxis_title="mg/dL",
                         yaxis_range=[min(80, float(out.min()) - 10), float(out.max()) + 30])
        st.plotly_chart(f3, width="stretch")
        m0, m1, m2 = st.columns(3)
        m0.metric("Peak, as usual", f"{out[0].max():.0f}")
        m1.metric("Peak, doctor's plan", f"{out[1].max():.0f}", f"{out[1].max() - out[0].max():+.0f}", delta_color="inverse")
        m2.metric("Peak, suggested", f"{out[2].max():.0f}", f"{out[2].max() - out[0].max():+.0f}", delta_color="inverse")
    nudge = (f"Hi {P.display_name.split()[0]}, your glucose is likely to rise after {usual_slot}. "
             f"Today, try {swap.lower()} instead of {usual.lower()} and take a 20-minute walk about 20 minutes after eating.")
    n1, n2 = st.columns([3, 1])
    note = n1.text_area("Message to patient (editable)", nudge, height=80)
    if n2.button("Send nudge to patient", width="stretch"):
        st.success("Nudge queued to the patient's app (demo only, nothing is sent).")
    if n2.button("Flag for tele-consult", width="stretch"):
        st.info("Added to tomorrow's tele-consult list (demo).")

with tab2:
    wk = pr.copy()
    wk["hour"] = wk.index.hour + wk.index.minute / 60
    agp = wk.groupby((wk.index.hour * 60 + wk.index.minute) // 30 * 30)["cgm_now"].quantile([0.05, 0.25, 0.5, 0.75, 0.95]).unstack()
    hrs = agp.index / 60
    f4 = go.Figure()
    f4.add_hrect(y0=70, y1=180, fillcolor=C[7], line_width=0, layer="below")
    f4.add_trace(go.Scatter(x=hrs, y=agp[0.95], line=dict(color=C[6], width=0), showlegend=False))
    f4.add_trace(go.Scatter(x=hrs, y=agp[0.05], fill="tonexty", fillcolor="rgba(143,184,232,0.35)", line=dict(width=0), name="5 to 95th pct"))
    f4.add_trace(go.Scatter(x=hrs, y=agp[0.75], line=dict(width=0), showlegend=False))
    f4.add_trace(go.Scatter(x=hrs, y=agp[0.25], fill="tonexty", fillcolor="rgba(47,109,181,0.45)", line=dict(width=0), name="25 to 75th pct"))
    f4.add_trace(go.Scatter(x=hrs, y=agp[0.5], line=dict(color=C[0], width=2.5), name="Median"))
    f4.update_layout(height=330, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", font=dict(family="Arial"),
                     title="Ambulatory glucose profile, monitoring week", xaxis_title="hour of day", yaxis_title="mg/dL")
    a, b = st.columns([1.6, 1])
    a.plotly_chart(f4, width="stretch")
    g = wk.cgm_raw.dropna()
    daily = wk.assign(d=wk.index.date).groupby("d").agg(
        mean=("cgm_raw", "mean"), tir=("cgm_raw", lambda x: 100 * ((x >= 70) & (x <= 180)).mean()),
        tar=("cgm_raw", lambda x: 100 * (x > 180).mean()))
    alerts = wk[(wk.p_spike >= THR) & wk.eligible]
    alerts = alerts[alerts.index.to_series().diff().fillna(pd.Timedelta("2h")) > pd.Timedelta("60min")]
    daily["alerts sent"] = alerts.groupby(alerts.index.date).size()
    daily = daily.fillna(0).round(0).astype(int)
    daily.index = [pd.Timestamp(d).strftime("%a %d") for d in daily.index]
    b.metric("Time in range 70 to 180", f"{100 * ((g >= 70) & (g <= 180)).mean():.0f}%", help="Target above 70%")
    b.metric("Time above 180", f"{100 * (g > 180).mean():.0f}%")
    b.metric("Glucose management indicator", f"{3.31 + 0.02392 * g.mean():.1f}%", help="CGM-estimated HbA1c")
    b.dataframe(daily.rename(columns={"mean": "mean mg/dL", "tir": "TIR %", "tar": "TAR %"}), width="stretch", height=290)

with tab3:
    w = ts[(ts.timestamp >= now - pd.Timedelta(hours=36)) & (ts.timestamp <= now)]
    f5 = make_subplots(rows=3, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                       subplot_titles=("Heart rate (bpm) and HRV RMSSD (ms)", "Steps per 5 min", "Sleep stage"))
    f5.add_trace(go.Scatter(x=w.timestamp, y=w.heart_rate_bpm, name="Heart rate", line=dict(color=C[1], width=1)), 1, 1)
    f5.add_trace(go.Scatter(x=w.timestamp, y=w.hrv_rmssd_ms, name="HRV", line=dict(color=C[5], width=1)), 1, 1)
    f5.add_trace(go.Bar(x=w.timestamp, y=w.steps, marker_color=C[3], name="Steps"), 2, 1)
    stg = w.sleep_stage.astype(str).map({"deep": 3, "light": 2, "rem": 1, "awake": 0})
    f5.add_trace(go.Scatter(x=w.timestamp, y=stg, line=dict(color=C[0], shape="hv"), name="Sleep"), 3, 1)
    f5.update_yaxes(tickvals=[0, 1, 2, 3], ticktext=["awake", "REM", "light", "deep"], row=3, col=1)
    f5.update_layout(height=520, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white", showlegend=False, font=dict(family="Arial"))
    st.plotly_chart(f5, width="stretch")

with tab4:
    a, b = st.columns([1, 1.2])
    a.dataframe(P.rename("value").astype(str), width="stretch", height=520)
    with b.expander("Raw HL7 FHIR R4 bundle (what the twin actually ingests)", expanded=False):
        st.json(json.loads((ROOT / f"data/ehr/fhir/{pid}.json").read_text()))
    b.markdown(f"**Model card.** LightGBM over {len(COLS)} fused features. Test AUROC "
               f"{METRICS['ablation_test']['GlucoSentinel: EHR + Wearables + Twin']['auroc']:.3f}. "
               f"Warns {METRICS['episodes_test_glucosentinel']['warned_30min_ahead_pct']:.0f}% of spikes 30+ minutes ahead, "
               f"median lead {METRICS['episodes_test_glucosentinel']['median_lead_min']:.0f} min, "
               f"{METRICS['episodes_test_glucosentinel']['false_alerts_per_day']} false alerts per patient-day. "
               "Trained and tested on synthetic data only; not a medical device.")

st.caption("GlucoSentinel is a research prototype built on synthetic data for the Happiest Health Digital Twin Challenge 2026. "
           "It is not a medical device and must not be used for clinical decisions.")
