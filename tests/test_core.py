"""Fast unit tests for the pieces the results depend on."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from glucosentinel import physiology as phy  # noqa: E402
from glucosentinel.features import labels, spike_episodes  # noqa: E402
from glucosentinel.fhir import read_bundle  # noqa: E402
from glucosentinel.twin import PatientTwin  # noqa: E402
from glucosentinel.train import clarke_zone  # noqa: E402

P = dict(V=110.0, SG=0.014, SI=3e-4, p2=0.025, n=0.12, gamma=0.05, Ib=8.0, ex=0.009)


def run(carbs_g, steps=0.0, minutes=300):
    s = phy.init_state(np.array([120.0]), P["Ib"])
    out = []
    for t in range(minutes):
        cf = carbs_g * 1000 * 0.7 if t == 10 else 0.0
        cs = carbs_g * 1000 * 0.3 if t == 10 else 0.0
        act = steps if 30 <= t < 50 else 0.0
        s = phy.step(s, P, cf, cs, act, 120.0, 1.0, 1.0)
        out.append(float(s["G"][0]))
    return np.array(out)


def test_fasting_is_steady():
    g = run(0)
    assert abs(g[-1] - 120) < 1.0


def test_meal_raises_then_returns():
    g = run(60)
    assert g.max() > 170
    assert g[-1] < g.max() - 40


def test_more_carbs_bigger_peak_and_walking_helps():
    assert run(90).max() > run(45).max()
    assert run(60, steps=100).max() < run(60).max()


def test_spike_labels_look_ahead_two_hours():
    cgm = np.full(100, 120.0)
    cgm[60:70] = 200.0                      # one 50-minute excursion
    onsets, _ = spike_episodes(cgm)
    assert list(onsets) == [60]
    y, eligible, _ = labels(cgm)
    assert y[36:60].all() and not y[:36].any()
    assert not eligible[60:70].any()        # no prediction while already above 180


def test_short_blips_are_not_spikes():
    cgm = np.full(50, 120.0)
    cgm[20:22] = 190.0
    assert len(spike_episodes(cgm)[0]) == 0


def test_clarke_zones():
    z = clarke_zone(np.array([100, 250, 50]), np.array([105, 100, 200]))
    assert list(z) == ["A", "D", "E"]


def test_fhir_round_trip_and_twin_prior():
    ehr = read_bundle(ROOT / "data/ehr/fhir/PT-0054.json")
    flat = pd.read_csv(ROOT / "data/ehr/patients.csv").set_index("patient_id").loc["PT-0054"]
    assert ehr["hba1c_pct"] == flat.hba1c_pct and ehr["fpg_mgdl"] == flat.fpg_mgdl
    tw = PatientTwin(ehr)
    assert 1e-5 < tw.prior["SI"] < 2e-3 and 80 < tw.prior["Gb"] < 240


def test_saved_metrics_match_claims():
    m = json.loads((ROOT / "reports/metrics.json").read_text())
    a = m["ablation_test"]
    assert a["GlucoSentinel: EHR + Wearables + Twin"]["auroc"] > a["EHR + Wearables (fusion)"]["auroc"]
    assert m["episodes_test_glucosentinel"]["false_alerts_per_day"] <= 1.0
