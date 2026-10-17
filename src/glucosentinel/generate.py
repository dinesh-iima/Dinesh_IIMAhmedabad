"""Step 1: generate the synthetic cohort, EHR (CSV + FHIR) and wearable streams.

    python -m glucosentinel.generate --n 120
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from .cohort import make_cohort
from .simulate import simulate_cohort, add_labs
from .fhir import write_bundles

ROOT = Path(__file__).resolve().parents[2]
EHR_COLS = ["patient_id", "display_name", "sex", "age", "region", "city", "diagnosis", "icd10",
            "diabetes_duration_y", "bmi", "waist_cm", "hypertension", "dyslipidaemia",
            "family_history_t2d", "tcf7l2_rs7903146_risk_alleles", "t2d_prs", "on_metformin",
            "on_sulfonylurea", "on_dpp4i", "resting_hr", "sbp", "dbp", "ldl", "hdl",
            "triglycerides", "egfr", "hba1c_pct", "hba1c_6m_ago", "hba1c_12m_ago", "fpg_mgdl"]


def main(n=120, seed=7):
    data = ROOT / "data"
    for sub in ("ehr/fhir", "wearables", "ground_truth"):
        (data / sub).mkdir(parents=True, exist_ok=True)
    cohort = make_cohort(n, seed=seed)
    ts, meal_log, meals_true, days = simulate_cohort(cohort, seed=seed + 4)
    cohort = add_labs(cohort, ts, seed=seed + 6)

    cohort[EHR_COLS].to_csv(data / "ehr/patients.csv", index=False)
    write_bundles(cohort, data / "ehr/fhir")
    ts.drop(columns=["_true_glucose"]).to_parquet(data / "wearables/cgm_wearables_5min.parquet", index=False)
    meal_log.to_csv(data / "wearables/meal_log.csv", index=False)

    # simulator-only files, used to validate the twin; never used as model inputs
    ts[["patient_id", "timestamp", "_true_glucose"]].rename(columns={"_true_glucose": "true_blood_glucose_mgdl"}) \
        .to_parquet(data / "ground_truth/true_glucose_5min.parquet", index=False)
    meals_true.to_csv(data / "ground_truth/meals_eaten.csv", index=False)
    days.to_csv(data / "ground_truth/daily_context.csv", index=False)
    hidden = {pid: dict(params=pp, lifestyle=ll) for pid, pp, ll in zip(cohort.patient_id, cohort._params, cohort._life)}
    (data / "ground_truth/hidden_physiology.json").write_text(json.dumps(hidden, indent=1, default=float))
    print(f"patients={n} rows={len(ts):,} meals_logged={len(meal_log):,} -> {data}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    main(a.n, a.seed)
