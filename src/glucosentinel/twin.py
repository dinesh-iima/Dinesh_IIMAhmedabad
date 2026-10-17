"""The patient digital twin.

A PatientTwin is a personalised copy of the physiology model in physiology.py.

1. PRIOR (from the EHR / FHIR record): population equations turn HbA1c, fasting glucose,
   BMI, diabetes duration, TCF7L2 genotype, PRS and medications into starting guesses for
   insulin sensitivity, beta-cell response, basal glucose and dawn effect.
2. CALIBRATION (from wearables): over a 7-day onboarding week the twin searches for the
   parameters that best reproduce the patient's own CGM response to the meals and walks it
   saw, with a penalty for straying far from the EHR prior.
3. ASSIMILATION (live): every 5 minutes the twin's state is nudged toward the latest CGM
   reading, the way a weather model assimilates new observations.
4. SIMULATION: from any moment the twin can be run forward under "what-if" scenarios
   (a different meal, a post-meal walk, a better night's sleep, a medication change).
"""
import json
import numpy as np
import pandas as pd

from . import physiology as phy
from .foods import fast_fraction

DT = 5.0           # minutes per twin step
H = 24             # 2-hour forecast horizon in steps
K_OBS = 0.6        # observer gain toward CGM
K_SLEEP = 0.22     # population effect of a poor night on next-day insulin sensitivity
FIXED = dict(SG=0.014, p2=0.025, n=0.12, ex=0.009)


def dawn_bump(hours):
    return np.exp(-0.5 * ((hours - 6.3) / 1.1) ** 2)


def ehr_prior(ehr):
    """Population-level mapping from EHR fields to physiological starting values."""
    height = 1.70 if ehr["sex"] == "M" else 1.57
    weight = ehr["bmi"] * height ** 2
    a1c = ehr["hba1c_pct"]
    SI = 4.6e-4 * np.exp(-0.03 * (ehr["bmi"] - 25) - 0.33 * (a1c - 6.3) - 0.05 * ehr.get("t2d_prs", 0.0))
    if ehr.get("on_metformin"):
        SI *= 1.10
    gamma = 0.068 * np.exp(-0.028 * ehr["diabetes_duration_y"] - 0.11 * ehr["tcf7l2_rs7903146_risk_alleles"]
                           - 0.10 * (a1c - 7))
    if ehr.get("on_sulfonylurea"):
        gamma *= 1.35
    if ehr.get("on_dpp4i"):
        gamma *= 1.15
    dawn = float(np.clip(6 + 4.0 * (a1c - 6.5), 2, 30))
    Gb = float(np.clip(ehr["fpg_mgdl"] - 0.75 * dawn, 85, 230))
    return dict(SI=float(SI), gamma=float(gamma), Gb=Gb, dawn=dawn,
                Ib=float(7.0 + 0.25 * (ehr["bmi"] - 22)), weight=float(weight), V=float(1.6 * weight))


def build_inputs(ts, meals, start, n_steps):
    """5-minute twin inputs from wearable and meal-log data (all observable, no ground truth)."""
    idx = pd.date_range(start, periods=n_steps, freq="5min")
    cf = np.zeros(n_steps); cs = np.zeros(n_steps)
    k = ((meals.timestamp - start) / pd.Timedelta("5min")).astype(int).to_numpy()
    ok = (k >= 0) & (k < n_steps)
    for kk, c, gi in zip(k[ok], meals.carbs_logged_g.to_numpy()[ok], meals.gi.to_numpy()[ok]):
        ff = fast_fraction(gi)
        cf[kk] += c * 1000 * ff
        cs[kk] += c * 1000 * (1 - ff)
    t = ts.set_index("timestamp").reindex(idx)
    cad = (t.steps.fillna(0).to_numpy() / 5.0)
    cgm = t.cgm_mgdl.to_numpy()
    hours = (idx.hour + idx.minute / 60.0).to_numpy()
    sleep_poor = nightly_poor_score(t)
    si_mult = 1 - K_SLEEP * sleep_poor
    return dict(idx=idx, carb_f=cf, carb_s=cs, cad=cad, cgm=cgm, hours=hours, si_mult=si_mult)


def nightly_poor_score(t):
    """Wearable-derived sleep-debt score (0 good to 1 poor) for the night before each day."""
    st = t.sleep_stage.astype(str)
    day = t.index.normalize()
    out = np.zeros(len(t))
    # a night belongs to the day you wake up on: window 18:00 previous day to 12:00
    shifted_day = (t.index + pd.Timedelta(hours=6)).normalize()
    asleep = st.isin(["light", "deep", "rem"]).to_numpy()
    in_bed_awake = (st == "awake").to_numpy()
    df = pd.DataFrame({"d": shifted_day, "asleep": asleep, "wake": in_bed_awake})
    agg = df.groupby("d").agg(sleep_steps=("asleep", "sum"), awake_steps=("wake", "sum"))
    hrs = agg.sleep_steps * 5 / 60.0
    frag = agg.awake_steps * 5 / 60.0
    score = (np.clip((7.0 - hrs) / 3.0, 0, 1) * 0.6 + np.clip(frag / 1.2, 0, 1) * 0.4)
    m = dict(zip(agg.index, score))
    out = np.array([m.get(d, 0.0) for d in day])
    return out


class PatientTwin:
    def __init__(self, ehr):
        self.ehr = ehr
        self.prior = ehr_prior(ehr)
        self.params = dict(self.prior)
        self.fit_info = {}

    # ------------------------------------------------------------------ core
    def _P(self, prm):
        P = dict(FIXED)
        P.update(V=prm["V"], Ib=prm["Ib"], SI=prm["SI"], gamma=prm["gamma"])
        return P

    def filter(self, inp, prm=None, batch=None):
        """Run the twin through history while assimilating CGM. Returns stacked states (8, [C,] T)."""
        prm = prm or self.params
        T = len(inp["cgm"])
        if batch is None:
            P = self._P(prm)
            Gb, dawn = prm["Gb"], prm["dawn"]
        else:
            P, Gb, dawn = batch
        cgm0 = inp["cgm"][~np.isnan(inp["cgm"])][0]
        s = phy.init_state(np.broadcast_to(cgm0, np.shape(Gb)).astype(float), P["Ib"])
        out = np.zeros((8,) + np.shape(Gb) + (T,))
        bump = dawn_bump(inp["hours"])
        for t in range(T):
            gb_t = Gb + dawn * bump[t]
            s = phy.step(s, P, inp["carb_f"][t], inp["carb_s"][t], inp["cad"][t], gb_t, inp["si_mult"][t], DT)
            c = inp["cgm"][t]
            if not np.isnan(c):
                s["G"] = s["G"] + K_OBS * (c - s["G"])
            out[..., t] = phy.stack(s)
        return out

    def forecast(self, states, inp, prm=None, batch=None, extra_carbs=None, extra_offset=None, horizon=H):
        """Forecast glucose for the next `horizon` steps from every stored state.

        states: (8, [C,] T). Returns (..., T, horizon). No future meals or activity are
        assumed unless extra_carbs (mg, fast-pool share applied) and extra_offset (steps) are given.
        """
        prm = prm or self.params
        if batch is None:
            P = self._P(prm); Gb, dawn = prm["Gb"], prm["dawn"]
        else:
            P, Gb, dawn = batch
        Pb = {k: (np.asarray(v)[..., None] if np.ndim(v) else v) for k, v in P.items()}
        Gb_b = np.asarray(Gb)[..., None] if np.ndim(Gb) else Gb
        dawn_b = np.asarray(dawn)[..., None] if np.ndim(dawn) else dawn
        s = phy.unstack(states.copy())
        T = states.shape[-1]
        hrs = inp["hours"]
        si = inp["si_mult"]
        res = np.zeros(states.shape[1:] + (horizon,))
        tt = np.arange(T)
        for h in range(horizon):
            j = np.minimum(tt + h + 1, T - 1)
            hr_future = (hrs[tt] + (h + 1) * DT / 60.0) % 24
            gb_t = Gb_b + dawn_b * dawn_bump(hr_future)
            cf = cs = 0.0
            if extra_carbs is not None:
                hit = (extra_offset == h)
                cf = np.where(hit, extra_carbs * 0.55, 0.0)
                cs = np.where(hit, extra_carbs * 0.45, 0.0)
            s = phy.step(s, Pb, cf, cs, 0.0, gb_t, si[j], DT)
            res[..., h] = s["G"]
        return res

    # ------------------------------------------------------------- calibrate
    def calibrate(self, inp, n_days=7, rounds=2, seed=0):
        """Fit SI, gamma, Gb, dawn to the onboarding week (penalised toward the EHR prior)."""
        T = int(n_days * 288)
        sub = {k: (v[:T] if isinstance(v, np.ndarray) else v) for k, v in inp.items()}
        cgm = sub["cgm"]
        center = dict(lSI=np.log(self.prior["SI"]), lg=np.log(self.prior["gamma"]),
                      Gb=self.prior["Gb"], dawn=self.prior["dawn"])
        spans = dict(lSI=1.0, lg=0.8, Gb=30.0, dawn=12.0)
        best = None
        for r in range(rounds):
            grids = [np.linspace(center[k] - spans[k], center[k] + spans[k], 5 if r == 0 else 4)
                     for k in ("lSI", "lg", "Gb", "dawn")]
            mesh = np.array(np.meshgrid(*grids, indexing="ij")).reshape(4, -1)
            lSI, lg, Gb, dawn = mesh
            dawn = np.clip(dawn, 0, 40)
            P = dict(FIXED); P.update(V=self.prior["V"], Ib=self.prior["Ib"], SI=np.exp(lSI), gamma=np.exp(lg))
            states = self.filter(sub, batch=(P, Gb, dawn))
            fc = self.forecast(states, sub, batch=(P, Gb, dawn))          # (C, T, H)
            err = 0.0
            for hstep in (6, 12, 24):
                target = np.full(T, np.nan); target[:T - hstep] = cgm[hstep:]
                d = fc[:, :, hstep - 1] - target[None, :]
                err = err + np.nanmean(d ** 2, axis=1)
            pen = (((lSI - np.log(self.prior["SI"])) / 0.6) ** 2 + ((lg - np.log(self.prior["gamma"])) / 0.6) ** 2
                   + ((Gb - self.prior["Gb"]) / 25) ** 2) * 20.0
            obj = err + pen
            b = int(np.argmin(obj))
            best = dict(lSI=lSI[b], lg=lg[b], Gb=Gb[b], dawn=dawn[b], rmse=float(np.sqrt(err[b] / 3)))
            center = {k: best[k] for k in ("lSI", "lg", "Gb", "dawn")}
            spans = {k: v * 0.35 for k, v in spans.items()}
        self.params.update(SI=float(np.exp(best["lSI"])), gamma=float(np.exp(best["lg"])),
                           Gb=float(best["Gb"]), dawn=float(best["dawn"]))
        self.fit_info = dict(calibration_days=n_days, forecast_rmse_mgdl=round(best["rmse"], 1))
        return self

    # --------------------------------------------------------------- what-if
    def what_if(self, state, hour, scenarios, hours_ahead=4.0):
        """Simulate several scenarios from one state. Each scenario is a dict with optional keys:
        meals: list of (offset_min, carbs_g, gi); walks: list of (offset_min, minutes, cadence);
        si_factor (sleep or medication effect on insulin sensitivity); gb_factor (hepatic output).
        Returns array (n_scenarios, steps) of glucose in mg/dL."""
        n = int(hours_ahead * 60 / DT)
        C = len(scenarios)
        s = phy.unstack(np.repeat(state[:, None], C, axis=1))
        P = self._P(self.params)
        cf = np.zeros((C, n)); cs = np.zeros((C, n)); cad = np.zeros((C, n))
        si = np.ones(C); gbf = np.ones(C)
        for c, sc in enumerate(scenarios):
            for off, carbs, gi in sc.get("meals", []):
                k = int(off // DT)
                if k < n:
                    ff = fast_fraction(gi)
                    cf[c, k] += carbs * 1000 * ff; cs[c, k] += carbs * 1000 * (1 - ff)
            for off, mins, cadence in sc.get("walks", []):
                k0, k1 = int(off // DT), int((off + mins) // DT)
                cad[c, k0:k1] = cadence
            si[c] = sc.get("si_factor", 1.0); gbf[c] = sc.get("gb_factor", 1.0)
        out = np.zeros((C, n))
        a_eff = np.zeros(C)
        for k in range(n):
            hr = (hour + (k + 1) * DT / 60.0) % 24
            gb_t = self.params["Gb"] * gbf + self.params["dawn"] * dawn_bump(hr)
            a_eff += ((cad[:, k] >= 80).astype(float) - a_eff) * DT / 120.0
            s = phy.step(s, P, cf[:, k], cs[:, k], cad[:, k], gb_t, si * (1 + 0.25 * a_eff), DT)
            out[:, k] = s["G"]
        return out

    # ----------------------------------------------------------------- io
    def to_json(self):
        return json.dumps(dict(ehr=self.ehr, prior=self.prior, params=self.params, fit=self.fit_info), indent=1, default=float)

    @classmethod
    def from_json(cls, txt):
        d = json.loads(txt)
        t = cls(d["ehr"]); t.prior = d["prior"]; t.params = d["params"]; t.fit_info = d["fit"]
        return t
