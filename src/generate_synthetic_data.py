"""
generate_synthetic_data.py
--------------------------
Generates 5 000 synthetic hunt-log records and writes them to
data/raw/hunt_logs.csv.  Each record simulates a single hunt session
from the Hunt Master Academy platform.

Columns produced
----------------
hunt_id        : UUID — unique session identifier
date           : random date within the current year
quarry         : whitetail | turkey | elk
lat            : latitude  (continental US bounding box)
lon            : longitude (continental US bounding box)
wind_speed     : knots, 0–25
wind_dir       : compass rose (N, NE, E, …, NW)
temp_c         : °C, -5 to 35
moon_phase     : new | waxing_crescent | first_quarter | waxing_gibbous |
                 full | waning_gibbous | last_quarter | waning_crescent
strategy_used  : ambush | still_hunt | stalking | calling
duration_min   : hunt duration in minutes (30–480)
success        : 0 or 1 (1 = harvested / tagged animal)
"""

import os
import random
from pathlib import Path

import polars as pl
from faker import Faker

SEED = 42
NUM_RECORDS = 5_000
OUTPUT_PATH = Path("data/raw/hunt_logs.csv")

fake = Faker()
Faker.seed(SEED)
random.seed(SEED)

QUARRY = ["whitetail", "turkey", "elk"]
WIND_DIRS = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
MOON_PHASES = [
    "new", "waxing_crescent", "first_quarter", "waxing_gibbous",
    "full", "waning_gibbous", "last_quarter", "waning_crescent",
]
STRATEGIES = ["ambush", "still_hunt", "stalking", "calling"]

# Per-quarry base success rates (drives realistic class imbalance)
SUCCESS_WEIGHTS: dict[str, list[float]] = {
    "whitetail": [0.62, 0.38],
    "turkey":    [0.55, 0.45],
    "elk":       [0.72, 0.28],
}


def _success_weight(quarry: str, wind_speed: float, strategy: str) -> list[float]:
    """Slightly adjust success probability based on conditions."""
    base = SUCCESS_WEIGHTS[quarry]
    # Calm wind favours ambush; high wind hurts calling
    if wind_speed < 8 and strategy == "ambush":
        base = [max(0.45, base[0] - 0.08), min(0.55, base[1] + 0.08)]
    if wind_speed > 18 and strategy == "calling":
        base = [min(0.80, base[0] + 0.10), max(0.20, base[1] - 0.10)]
    return base


def generate(n: int = NUM_RECORDS) -> pl.DataFrame:
    records = []
    for _ in range(n):
        quarry = random.choice(QUARRY)
        wind_speed = round(random.uniform(0, 25), 1)
        strategy = random.choice(STRATEGIES)
        weights = _success_weight(quarry, wind_speed, strategy)
        records.append(
            {
                "hunt_id":      fake.uuid4(),
                "date":         str(fake.date_this_year()),
                "quarry":       quarry,
                "lat":          round(random.uniform(30.0, 49.0), 5),
                "lon":          round(random.uniform(-120.0, -70.0), 5),
                "wind_speed":   wind_speed,
                "wind_dir":     random.choice(WIND_DIRS),
                "temp_c":       round(random.uniform(-5, 35), 1),
                "moon_phase":   random.choice(MOON_PHASES),
                "strategy_used": strategy,
                "duration_min": random.randint(30, 480),
                "success":      random.choices([0, 1], weights=weights)[0],
            }
        )

    return pl.DataFrame(records)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = generate()
    df.write_csv(OUTPUT_PATH)
    print(f"✅ Generated {len(df)} records → {OUTPUT_PATH}")
    print(df.head(3))


if __name__ == "__main__":
    main()
