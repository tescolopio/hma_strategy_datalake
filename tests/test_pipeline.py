"""
tests/test_pipeline.py
----------------------
End-to-end smoke tests for the HMA Strategy Data Lake pipeline.

Covers:
  - Synthetic data generation (schema & row count)
  - Ingest to Bronze Delta Lake
  - ETL Bronze → Silver → Gold
  - Model training
  - AI Coach prediction
"""

import pathlib
import sys
import tempfile

import polars as pl
import pytest

# Allow importing from src/
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent))

from src.generate_synthetic_data import generate
from src.etl import bronze_to_silver, silver_to_gold
from src.ingest import ingest
from src.train_model import FEATURES, TARGET, train
from src.predict_strategy import coach_recommend


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def tmp_dir(tmp_path_factory):
    return tmp_path_factory.mktemp("lake")


@pytest.fixture(scope="module")
def raw_csv(tmp_dir):
    """Generate a small CSV file for testing."""
    csv_path = tmp_dir / "hunt_logs.csv"
    df = generate(n=200)
    df.write_csv(csv_path)
    return csv_path


@pytest.fixture(scope="module")
def bronze_path(tmp_dir, raw_csv):
    path = str(tmp_dir / "bronze" / "hunts")
    ingest(csv_path=raw_csv, delta_path=path)
    return path


@pytest.fixture(scope="module")
def silver_path(tmp_dir, bronze_path):
    path = str(tmp_dir / "silver" / "hunts")
    bronze_to_silver(bronze_path=bronze_path, silver_path=path)
    return path


@pytest.fixture(scope="module")
def gold_path(tmp_dir, silver_path):
    path = str(tmp_dir / "gold" / "hunts_features")
    silver_to_gold(silver_path=silver_path, gold_path=path)
    return path


@pytest.fixture(scope="module")
def trained_model(tmp_dir, gold_path):
    model_path = tmp_dir / "strategy_model.pkl"
    feat_path  = tmp_dir / "feature_names.txt"
    model = train(
        gold_path=gold_path,
        model_path=model_path,
        feat_path=feat_path,
        n_estimators=10,
    )
    return model, model_path


# ─────────────────────────────────────────────────────────────────────────────
# Tests
# ─────────────────────────────────────────────────────────────────────────────

class TestDataGeneration:
    def test_row_count(self):
        df = generate(n=100)
        assert len(df) == 100

    def test_schema_columns(self):
        df = generate(n=50)
        expected = {
            "hunt_id", "date", "quarry", "lat", "lon",
            "wind_speed", "wind_dir", "temp_c", "moon_phase",
            "strategy_used", "duration_min", "success",
        }
        assert expected.issubset(set(df.columns))

    def test_success_binary(self):
        df = generate(n=200)
        assert set(df["success"].unique().to_list()).issubset({0, 1})

    def test_wind_speed_range(self):
        df = generate(n=200)
        assert df["wind_speed"].min() >= 0
        assert df["wind_speed"].max() <= 25


class TestIngest:
    def test_bronze_row_count(self, bronze_path):
        from deltalake import DeltaTable
        dt = DeltaTable(bronze_path)
        df = pl.from_arrow(dt.to_pyarrow_table())
        assert len(df) == 200

    def test_bronze_schema(self, bronze_path):
        from deltalake import DeltaTable
        dt = DeltaTable(bronze_path)
        df = pl.from_arrow(dt.to_pyarrow_table())
        assert "hunt_id" in df.columns
        assert "success" in df.columns


class TestETL:
    def test_silver_row_count(self, silver_path):
        from deltalake import DeltaTable
        df = pl.from_arrow(DeltaTable(silver_path).to_pyarrow_table())
        assert len(df) > 0

    def test_silver_no_duplicates(self, silver_path):
        from deltalake import DeltaTable
        df = pl.from_arrow(DeltaTable(silver_path).to_pyarrow_table())
        assert not df.is_duplicated().any()

    def test_gold_has_feature_columns(self, gold_path):
        from deltalake import DeltaTable
        df = pl.from_arrow(DeltaTable(gold_path).to_pyarrow_table())
        for col in ["wind_favorable", "is_ambush", "is_whitetail", "moon_numeric"]:
            assert col in df.columns, f"Missing column: {col}"


class TestTraining:
    def test_model_saved(self, trained_model):
        _, model_path = trained_model
        assert model_path.exists()

    def test_model_has_features(self, trained_model):
        model, _ = trained_model
        assert hasattr(model, "feature_importances_")
        assert len(model.feature_importances_) == len(FEATURES)

    def test_model_accuracy_above_chance(self, trained_model, gold_path):
        import pandas as pd
        from deltalake import DeltaTable
        model, _ = trained_model
        df = pd.DataFrame(DeltaTable(gold_path).to_pyarrow_table().to_pydict())
        X, y = df[FEATURES], df[TARGET]
        acc = model.score(X, y)
        assert acc > 0.50, f"Model accuracy {acc:.1%} is not better than chance"


class TestPredict:
    def test_coach_recommend_keys(self, trained_model):
        model, model_path = trained_model
        rec = coach_recommend(
            quarry="whitetail",
            wind_speed=5,
            wind_dir="NW",
            temp_c=10,
            duration_min=180,
            moon_phase="waxing_crescent",
            model_path=model_path,
        )
        assert "recommended_strategy" in rec
        assert "success_probability" in rec
        assert "all_strategies" in rec

    def test_coach_four_strategies(self, trained_model):
        model, model_path = trained_model
        rec = coach_recommend(model_path=model_path)
        assert len(rec["all_strategies"]) == 4

    def test_coach_probability_range(self, trained_model):
        model, model_path = trained_model
        rec = coach_recommend(model_path=model_path)
        for s in rec["all_strategies"]:
            assert 0.0 <= s["success_probability"] <= 1.0
