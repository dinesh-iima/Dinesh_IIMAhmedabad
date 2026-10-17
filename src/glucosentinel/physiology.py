"""Glucose-insulin physiology shared by the data simulator and the digital twin.

The core is the Bergman minimal model extended with
  * a two-pool gut absorption model (fast and slow carbohydrates, driven by meal GI),
  * exercise-driven glucose uptake (from step cadence),
  * a time-varying basal glucose level (dawn phenomenon, stress, hepatic output),
  * a time-varying insulin-sensitivity multiplier (sleep debt, stress, recent activity),
  * a first-order interstitial lag so the twin speaks the same language as a CGM.

Everything is vectorised over a leading batch dimension so the same code simulates
120 patients at once (data generation), 200 candidate parameter sets at once
(twin calibration) or 4,000 forecast start points at once (twin forecasting).

Units: glucose mg/dL, insulin uU/mL, time minutes, carbohydrate input mg.
"""
import numpy as np

T_FAST = 30.0      # min, time-to-peak of fast-absorbing carbohydrate pool
T_SLOW = 65.0      # min, slow pool (fibre, protein, fat, low GI)
AG = 0.8           # bioavailable fraction of ingested carbohydrate
TAU_ISF = 8.0      # min, blood to interstitial fluid lag seen by a CGM
STATE_KEYS = ("G", "X", "I", "Q1f", "Q2f", "Q1s", "Q2s", "Gi")


def init_state(G0, Ib):
    G0 = np.asarray(G0, dtype=float)
    z = np.zeros_like(G0)
    return {"G": G0.copy(), "X": z.copy(), "I": np.broadcast_to(Ib, G0.shape).astype(float).copy(),
            "Q1f": z.copy(), "Q2f": z.copy(), "Q1s": z.copy(), "Q2s": z.copy(), "Gi": G0.copy()}


def step(s, p, carb_f, carb_s, act, gb_t, si_mult, dt):
    """One explicit Euler step. All arguments broadcast over the batch dimension.

    p must contain V (dL), SG, SI, p2, n, gamma, Ib, ex.
    carb_f / carb_s are carbohydrate amounts (mg) eaten during this step.
    act is step cadence (steps/min). gb_t is the basal glucose target (mg/dL).
    """
    G, X, I = s["G"], s["X"], s["I"]
    Q1f, Q2f, Q1s, Q2s = s["Q1f"], s["Q2f"], s["Q1s"], s["Q2s"]
    Ra = AG * (Q2f / T_FAST + Q2s / T_SLOW) / p["V"]
    dG = (-(p["SG"] + X) * G + p["SG"] * gb_t + Ra
          - p["ex"] * (act / 100.0) * np.maximum(G - 60.0, 0.0))
    dX = -p["p2"] * X + p["p2"] * p["SI"] * si_mult * (I - p["Ib"])
    dI = -p["n"] * (I - p["Ib"]) + p["gamma"] * np.maximum(G - gb_t, 0.0)
    return {
        "Q1f": Q1f + dt * (-Q1f / T_FAST) + carb_f,
        "Q2f": Q2f + dt * (Q1f - Q2f) / T_FAST,
        "Q1s": Q1s + dt * (-Q1s / T_SLOW) + carb_s,
        "Q2s": Q2s + dt * (Q1s - Q2s) / T_SLOW,
        "G": np.clip(G + dt * dG, 35.0, 600.0),
        "X": np.maximum(X + dt * dX, 0.0),
        "I": np.maximum(I + dt * dI, 0.0),
        "Gi": s["Gi"] + dt * (G - s["Gi"]) / TAU_ISF,
    }


def stack(s):
    return np.stack([s[k] for k in STATE_KEYS], axis=0)


def unstack(a):
    return {k: a[i] for i, k in enumerate(STATE_KEYS)}
