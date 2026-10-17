"""Step 2: ingest EHR (FHIR) + wearables for every patient, calibrate each digital twin on its
onboarding week, and write the fused feature table.

    python -m glucosentinel.build_features
"""
import json
import time
from multiprocessing import Pool
from pathlib import Path

import pandas as pd

from .features import build_patient
from .fhir import read_bundle
from .store import write_by_patient

ROOT = Path(__file__).resolve().parents[2]


def _one(args):
    pid, ts_p, meals_p = args
    ehr = read_bundle(ROOT / f"data/ehr/fhir/{pid}.json")
    df, twin, _, onsets = build_patient(pid, ts_p, meals_p, ehr)
    return pid, df, twin.to_json()


def main(workers=2):
    t0 = time.time()
    ts = pd.read_parquet(ROOT / "data/wearables/cgm_wearables_5min.parquet")
    meals = pd.read_csv(ROOT / "data/wearables/meal_log.csv", parse_dates=["timestamp"])
    jobs = [(pid, g.reset_index(drop=True), meals[meals.patient_id == pid]) for pid, g in ts.groupby("patient_id")]
    out_tw = ROOT / "models/twins"; out_tw.mkdir(parents=True, exist_ok=True)
    frames = []
    with Pool(workers) as pool:
        for k, (pid, df, tj) in enumerate(pool.imap_unordered(_one, jobs)):
            frames.append(df)
            (out_tw / f"{pid}.json").write_text(tj)
            if (k + 1) % 20 == 0:
                print(f"  {k + 1}/{len(jobs)} twins calibrated ({time.time() - t0:.0f}s)")
    feats = pd.concat(frames, ignore_index=True).sort_values(["patient_id", "timestamp"])
    write_by_patient(feats, ROOT / "data/processed/features")
    print(f"features {feats.shape} in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
