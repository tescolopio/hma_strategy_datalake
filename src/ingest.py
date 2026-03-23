"""
ingest.py
---------
Reads the raw hunt-log CSV from data/raw/ and writes it to the
Bronze layer of the Delta Lake (lake/bronze/hunts).

The Bronze layer is an exact copy of the source data — no
transformations are applied.  Schema is enforced by Delta.

Usage
-----
    python -m src.ingest
"""

from pathlib import Path

import polars as pl
from deltalake import write_deltalake

RAW_CSV   = Path("data/raw/hunt_logs.csv")
BRONZE_PATH = "lake/bronze/hunts"


def ingest(
    csv_path: Path = RAW_CSV,
    delta_path: str = BRONZE_PATH,
    mode: str = "overwrite",
) -> None:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"Raw data not found at {csv_path}. "
            "Run `python -m src.generate_synthetic_data` first."
        )

    df = pl.read_csv(csv_path)
    print(f"📥 Read {len(df)} rows from {csv_path}")

    write_deltalake(delta_path, df.to_arrow(), mode=mode)
    print(f"✅ Ingested to Bronze Delta Lake → {delta_path}")
    print(f"   Schema: {df.schema}")


if __name__ == "__main__":
    ingest()
