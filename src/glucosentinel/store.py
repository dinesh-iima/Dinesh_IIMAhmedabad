"""Small parquet helpers: big tables are stored one file per patient so every file stays small
(easy to upload to GitHub through the browser) and the dashboard can load one patient at a time."""
import shutil
from pathlib import Path

import pandas as pd


def write_by_patient(df, folder):
    folder = Path(folder)
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    for pid, g in df.groupby("patient_id", sort=True):
        g.to_parquet(folder / f"{pid}.parquet", index=False, compression="zstd")


def read_by_patient(folder, pid=None, columns=None):
    folder = Path(folder)
    files = [folder / f"{pid}.parquet"] if pid else sorted(folder.glob("*.parquet"))
    return pd.concat([pd.read_parquet(f, columns=columns) for f in files], ignore_index=True)
