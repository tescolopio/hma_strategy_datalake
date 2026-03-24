"""
predict_strategy.py
-------------------
AI Coach inference module for Hunt Master Academy.

Given current hunt conditions, the trained RandomForest model returns:
  • success_probability for each candidate strategy
  • the recommended strategy (highest predicted probability)
  • top SHAP drivers explaining the recommendation

Usage (CLI)
-----------
    python -m src.predict_strategy \
        --quarry whitetail \
        --wind_speed 5 \
        --wind_dir NW \
        --temp_c 12 \
        --duration_min 180 \
        --moon_phase waxing_crescent

Usage (API)
-----------
    from src.predict_strategy import coach_recommend
    result = coach_recommend(quarry="whitetail", wind_speed=5, temp_c=12,
                             wind_dir="NW", duration_min=180,
                             moon_phase="waxing_crescent")
    print(result)
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

MODEL_PATH = Path("models/strategy_model.pkl")
FEAT_PATH  = Path("models/feature_names.txt")

STRATEGIES = ["ambush", "still_hunt", "stalking", "calling"]
MOON_ORDER = [
    "new", "waxing_crescent", "first_quarter", "waxing_gibbous",
    "full", "waning_gibbous", "last_quarter", "waning_crescent",
]
WIND_ORDER = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]


def _build_row(
    quarry: str,
    wind_speed: float,
    wind_dir: str,
    temp_c: float,
    duration_min: int,
    moon_phase: str,
    strategy: str,
) -> dict[str, Any]:
    return {
        "wind_speed":      wind_speed,
        "temp_c":          temp_c,
        "duration_min":    duration_min,
        "wind_favorable":  int(wind_speed < 10),
        "is_ambush":       int(strategy == "ambush"),
        "is_still_hunt":   int(strategy == "still_hunt"),
        "is_stalking":     int(strategy == "stalking"),
        "is_calling":      int(strategy == "calling"),
        "is_whitetail":    int(quarry == "whitetail"),
        "is_turkey":       int(quarry == "turkey"),
        "is_elk":          int(quarry == "elk"),
        "moon_numeric":    MOON_ORDER.index(moon_phase) if moon_phase in MOON_ORDER else 0,
        "wind_dir_numeric": WIND_ORDER.index(wind_dir) if wind_dir in WIND_ORDER else 0,
    }


def coach_recommend(
    quarry: str = "whitetail",
    wind_speed: float = 5.0,
    wind_dir: str = "NW",
    temp_c: float = 12.0,
    duration_min: int = 180,
    moon_phase: str = "waxing_crescent",
    model_path: Path = MODEL_PATH,
) -> dict[str, Any]:
    """Return AI Coach recommendation for given conditions."""
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path}. "
            "Run `python -m src.train_model` first."
        )

    model = joblib.load(model_path)
    feature_names = FEAT_PATH.read_text().splitlines() if FEAT_PATH.exists() else None

    results: list[dict[str, Any]] = []
    for strategy in STRATEGIES:
        row = _build_row(
            quarry=quarry,
            wind_speed=wind_speed,
            wind_dir=wind_dir,
            temp_c=temp_c,
            duration_min=duration_min,
            moon_phase=moon_phase,
            strategy=strategy,
        )
        X = pd.DataFrame([row])
        if feature_names:
            X = X[feature_names]
        prob = model.predict_proba(X)[0][1]
        results.append({"strategy": strategy, "success_probability": round(float(prob), 4)})

    results.sort(key=lambda r: r["success_probability"], reverse=True)
    best = results[0]

    # SHAP explanation for top strategy (optional — gracefully skipped if shap unavailable)
    shap_drivers: list[str] = []
    try:
        import shap  # noqa: PLC0415

        row_best = _build_row(
            quarry=quarry,
            wind_speed=wind_speed,
            wind_dir=wind_dir,
            temp_c=temp_c,
            duration_min=duration_min,
            moon_phase=moon_phase,
            strategy=best["strategy"],
        )
        X_best = pd.DataFrame([row_best])
        if feature_names:
            X_best = X_best[feature_names]
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_best)
        # shap_values shape: (n_classes, n_samples, n_features) for RF
        vals = shap_values[1][0] if isinstance(shap_values, list) else shap_values[0]
        feat_names = feature_names or list(X_best.columns)
        top_idx = np.argsort(np.abs(vals))[::-1][:3]
        shap_drivers = [f"{feat_names[i]} ({vals[i]:+.3f})" for i in top_idx]
    except Exception:  # noqa: BLE001
        shap_drivers = []

    recommendation = {
        "quarry":              quarry,
        "conditions": {
            "wind_speed":  wind_speed,
            "wind_dir":    wind_dir,
            "temp_c":      temp_c,
            "moon_phase":  moon_phase,
            "duration_min": duration_min,
        },
        "recommended_strategy": best["strategy"],
        "success_probability":  best["success_probability"],
        "all_strategies":       results,
        "top_shap_drivers":     shap_drivers,
    }
    return recommendation


def _print_recommendation(rec: dict[str, Any]) -> None:
    print("\n🤖 Hunt Master AI Coach Recommendation")
    print("=" * 45)
    print(f"  Quarry    : {rec['quarry']}")
    cond = rec["conditions"]
    print(f"  Wind      : {cond['wind_speed']} kn {cond['wind_dir']}")
    print(f"  Temp      : {cond['temp_c']} °C")
    print(f"  Moon      : {cond['moon_phase']}")
    print(f"  Duration  : {cond['duration_min']} min planned")
    print()
    print(f"  ✅ Best strategy : {rec['recommended_strategy'].upper()}")
    print(f"  🎯 Success prob  : {rec['success_probability']:.0%}")
    print()
    print("  Strategy breakdown:")
    for s in rec["all_strategies"]:
        bar = "█" * int(s["success_probability"] * 20)
        print(f"    {s['strategy']:12s} {s['success_probability']:.0%}  {bar}")
    if rec["top_shap_drivers"]:
        print()
        print("  Top factors driving recommendation:")
        for d in rec["top_shap_drivers"]:
            print(f"    • {d}")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hunt Master AI Coach — strategy recommendation"
    )
    parser.add_argument("--quarry",       default="whitetail",
                        choices=["whitetail", "turkey", "elk"])
    parser.add_argument("--wind_speed",   type=float, default=5.0)
    parser.add_argument("--wind_dir",     default="NW", choices=WIND_ORDER)
    parser.add_argument("--temp_c",       type=float, default=12.0)
    parser.add_argument("--duration_min", type=int,   default=180)
    parser.add_argument("--moon_phase",   default="waxing_crescent",
                        choices=MOON_ORDER)
    args = parser.parse_args()

    rec = coach_recommend(
        quarry=args.quarry,
        wind_speed=args.wind_speed,
        wind_dir=args.wind_dir,
        temp_c=args.temp_c,
        duration_min=args.duration_min,
        moon_phase=args.moon_phase,
    )
    _print_recommendation(rec)


if __name__ == "__main__":
    main()
