"""
etl.py
------
Medallion ETL pipeline:

  Bronze (raw)  →  Silver (cleaned / validated)  →  Gold (ML-ready features)

Silver transformations
----------------------
- Drop exact duplicate rows
- Filter rows with valid lat/lon (continental US bounding box)
- Filter realistic weather values (wind_speed 0–25, temp_c -5–35)
- Cast date column to Date type

Gold feature engineering
------------------------
- wind_favorable   : bool  — wind_speed < 10 knots
- is_ambush        : int   — strategy_used == "ambush"
- is_still_hunt    : int   — strategy_used == "still_hunt"
- is_stalking      : int   — strategy_used == "stalking"
- is_calling       : int   — strategy_used == "calling"
- is_whitetail     : int   — quarry == "whitetail"
- is_turkey        : int   — quarry == "turkey"
- is_elk           : int   — quarry == "elk"
- moon_numeric     : int   — ordinal encoding of moon phase (0–7)
- wind_dir_numeric : int   — ordinal encoding of wind direction (0–7)

Usage
-----
    python -m src.etl
"""

from pathlib import Path

import polars as pl
from deltalake import write_deltalake
from deltalake import DeltaTable

BRONZE_PATH = "lake/bronze/hunts"
SILVER_PATH = "lake/silver/hunts"
GOLD_PATH   = "lake/gold/hunts_features"

MOON_ORDER = [
    "new", "waxing_crescent", "first_quarter", "waxing_gibbous",
    "full", "waning_gibbous", "last_quarter", "waning_crescent",
]
WIND_ORDER = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def _load_delta(path: str) -> pl.DataFrame:
    return pl.from_arrow(DeltaTable(path).to_pyarrow_table())


def bronze_to_silver(
    bronze_path: str = BRONZE_PATH,
    silver_path: str = SILVER_PATH,
    mode: str = "overwrite",
) -> pl.DataFrame:
    df = _load_delta(bronze_path)
    print(f"🔶 Bronze rows:  {len(df)}")

    df = (
        df
        .unique()
        .filter(
            (pl.col("lat").is_between(24.0, 50.0))
            & (pl.col("lon").is_between(-125.0, -66.0))
            & (pl.col("wind_speed").is_between(0, 25))
            & (pl.col("temp_c").is_between(-5, 35))
        )
        .with_columns(pl.col("date").str.to_date("%Y-%m-%d"))
    )

    print(f"🥈 Silver rows:  {len(df)}  (after cleaning)")
    write_deltalake(silver_path, df.to_arrow(), mode=mode)
    print(f"✅ Silver written → {silver_path}")
    return df


def silver_to_gold(
    silver_path: str = SILVER_PATH,
    gold_path:   str = GOLD_PATH,
    mode: str = "overwrite",
) -> pl.DataFrame:
    df = _load_delta(silver_path)

    moon_map = {phase: i for i, phase in enumerate(MOON_ORDER)}
    wind_map = {d: i for i, d in enumerate(WIND_ORDER)}

    df = df.with_columns(
        # Wind / weather features
        (pl.col("wind_speed") < 10).cast(pl.Int8).alias("wind_favorable"),
        # Strategy one-hot
        (pl.col("strategy_used") == "ambush").cast(pl.Int8).alias("is_ambush"),
        (pl.col("strategy_used") == "still_hunt").cast(pl.Int8).alias("is_still_hunt"),
        (pl.col("strategy_used") == "stalking").cast(pl.Int8).alias("is_stalking"),
        (pl.col("strategy_used") == "calling").cast(pl.Int8).alias("is_calling"),
        # Quarry one-hot
        (pl.col("quarry") == "whitetail").cast(pl.Int8).alias("is_whitetail"),
        (pl.col("quarry") == "turkey").cast(pl.Int8).alias("is_turkey"),
        (pl.col("quarry") == "elk").cast(pl.Int8).alias("is_elk"),
        # Ordinal encodings
        pl.col("moon_phase").replace_strict(moon_map, default=0).cast(pl.Int8).alias("moon_numeric"),
        pl.col("wind_dir").replace_strict(wind_map, default=0).cast(pl.Int8).alias("wind_dir_numeric"),
    )

    print(f"🥇 Gold rows:    {len(df)}  (features engineered)")
    write_deltalake(gold_path, df.to_arrow(), mode=mode)
    print(f"✅ Gold written  → {gold_path}")
    return df


def run_pipeline() -> None:
    bronze_to_silver()
    silver_to_gold()
    print("\n🏁 Medallion ETL pipeline complete.")


if __name__ == "__main__":
    run_pipeline()
