"""Rebuild everything from scratch: synthetic data, FHIR bundles, 120 calibrated twins,
models, metrics and figures. About 8 minutes on a 2-core laptop.

    python run_all.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from glucosentinel import generate, build_features, train, figures  # noqa: E402

if __name__ == "__main__":
    t0 = time.time()
    print("[1/4] generating synthetic EHR (FHIR) and wearable streams")
    generate.main(n=120, seed=7)
    print("[2/4] calibrating 120 digital twins and building fused features")
    build_features.main()
    print("[3/4] training spike model and forecasters, evaluating on unseen patients")
    train.main()
    print("[4/4] drawing figures")
    figures.main()
    print(f"done in {time.time() - t0:.0f}s. Metrics: reports/metrics.json. Dashboard: streamlit run dashboard/app.py")
