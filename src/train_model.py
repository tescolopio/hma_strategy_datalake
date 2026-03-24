"""
train_model.py
--------------
Trains a RandomForestClassifier on the Gold layer feature table and
saves the fitted model to models/strategy_model.pkl.

Features used
-------------
wind_speed, temp_c, duration_min, wind_favorable,
is_ambush, is_still_hunt, is_stalking, is_calling,
is_whitetail, is_turkey, is_elk,
moon_numeric, wind_dir_numeric

Target
------
success  (binary: 0 = unsuccessful, 1 = successful harvest)

Outputs
-------
models/strategy_model.pkl   — serialised RandomForest model
models/feature_names.txt    — ordered list of feature names used during training

Usage
-----
    python -m src.train_model
"""

from pathlib import Path

import joblib
import pandas as pd
from deltalake import DeltaTable
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

GOLD_PATH   = "lake/gold/hunts_features"
MODEL_PATH  = Path("models/strategy_model.pkl")
FEAT_PATH   = Path("models/feature_names.txt")

FEATURES = [
    "wind_speed", "temp_c", "duration_min", "wind_favorable",
    "is_ambush", "is_still_hunt", "is_stalking", "is_calling",
    "is_whitetail", "is_turkey", "is_elk",
    "moon_numeric", "wind_dir_numeric",
]
TARGET = "success"


def train(
    gold_path: str = GOLD_PATH,
    model_path: Path = MODEL_PATH,
    feat_path: Path = FEAT_PATH,
    n_estimators: int = 200,
    random_state: int = 42,
    test_size: float = 0.2,
) -> RandomForestClassifier:
    df = pd.DataFrame(DeltaTable(gold_path).to_pyarrow_table().to_pydict())
    print(f"📊 Gold rows loaded: {len(df)}")

    X = df[FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    acc = model.score(X_test, y_test)
    print(f"\n✅ Model trained — Test accuracy: {acc:.1%}")
    print("\n📋 Classification report:")
    print(classification_report(y_test, model.predict(X_test)))

    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    feat_path.write_text("\n".join(FEATURES))
    print(f"💾 Model saved → {model_path}")
    print(f"💾 Features   → {feat_path}")

    return model


if __name__ == "__main__":
    train()
