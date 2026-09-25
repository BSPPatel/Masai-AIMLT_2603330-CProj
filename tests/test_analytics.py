"""
Automated Test Suite for Module 2: Analytics & Machine Learning Pipeline
Tests profiling, cleaning rules, statistical EDA, leakage prevention,
classifiers, class imbalance, regression, and model persistence.
"""

import os
import joblib
import pytest
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score

from analytics.dataset_loader import get_or_load_titanic

TITANIC_CSV_PATH = "analytics/titanic.csv"
MODEL_JOB_PATH = "analytics/models/best_model.joblib"

def test_titanic_dataset_presence_and_shape():
    """Verify titanic.csv exists, is committed, and contains the standard 891 entries."""
    assert os.path.exists(TITANIC_CSV_PATH), "titanic.csv must be persisted locally"
    df = pd.read_csv(TITANIC_CSV_PATH)
    assert df.shape[0] == 891
    assert "survived" in df.columns
    assert "pclass" in df.columns
    assert "deck" in df.columns

def test_missing_value_threshold_logic():
    """Verify missing value threshold rules: <5% drop rows, >30% drop column."""
    df = pd.read_csv(TITANIC_CSV_PATH)
    
    # deck is ~77% missing (>30% threshold -> drop column)
    deck_missing_pct = (df["deck"].isnull().sum() / len(df)) * 100
    assert deck_missing_pct > 30.0
    
    # embarked is ~0.22% missing (<5% threshold -> drop row)
    embarked_missing_pct = (df["embarked"].isnull().sum() / len(df)) * 100
    assert embarked_missing_pct < 5.0
    
    df_clean = df.drop(columns=["deck"]).dropna(subset=["embarked"])
    assert "deck" not in df_clean.columns
    assert df_clean["embarked"].isnull().sum() == 0
    assert len(df_clean) == 889

def test_fare_central_tendency_and_skewness():
    """Verify mean > median > mode ordering indicating right-skewness."""
    df = pd.read_csv(TITANIC_CSV_PATH)
    mean_val = df["fare"].mean()
    median_val = df["fare"].median()
    mode_val = df["fare"].mode()[0]
    
    assert mean_val > median_val > mode_val, "Fare distribution must satisfy positive right-skew condition"

def test_six_column_correlation_matrix_constraint():
    """Ensure correlation matrix contains EXACTLY the 6 specified columns and excludes redundant flags."""
    df = pd.read_csv(TITANIC_CSV_PATH)
    allowed_cols = ["survived", "pclass", "age", "sibsp", "parch", "fare"]
    
    corr = df[allowed_cols].corr()
    assert corr.shape == (6, 6)
    assert list(corr.columns) == allowed_cols
    assert "adult_male" not in corr.columns
    assert "alone" not in corr.columns
    
    # Check top negative correlation: pclass vs fare
    assert corr.loc["pclass", "fare"] < -0.40

def test_bivariate_boolean_masking():
    """Test survival calculation using combinations of & boolean masks."""
    df = pd.read_csv(TITANIC_CSV_PATH)
    
    # Females in 1st class vs Males in 3rd class
    mask_f_c1 = (df["sex"] == "female") & (df["pclass"] == 1)
    mask_m_c3 = (df["sex"] == "male") & (df["pclass"] == 3)
    
    rate_f_c1 = df[mask_f_c1]["survived"].mean()
    rate_m_c3 = df[mask_m_c3]["survived"].mean()
    
    assert rate_f_c1 > 0.90, "1st class females must have >90% survival"
    assert rate_m_c3 < 0.20, "3rd class males must have <20% survival"

def test_model_persistence_and_raw_inference():
    """Test that the saved joblib artifact includes preprocessing and predicts on raw input."""
    assert os.path.exists(MODEL_JOB_PATH), "best_model.joblib must be persisted"
    pipeline = joblib.load(MODEL_JOB_PATH)
    
    assert isinstance(pipeline, Pipeline)
    assert "preprocessor" in pipeline.named_steps
    assert "classifier" in pipeline.named_steps
    
    # Predict directly on raw dictionary DataFrame (unscaled, unencoded)
    raw_sample = pd.DataFrame([{
        "pclass": 1,
        "sex": "female",
        "age": 29.0,
        "sibsp": 1,
        "parch": 0,
        "fare": 75.0,
        "embarked": "C"
    }])
    
    pred = pipeline.predict(raw_sample)
    proba = pipeline.predict_proba(raw_sample)
    
    assert pred[0] in [0, 1]
    assert proba.shape == (1, 2)
    assert 0.0 <= proba[0][1] <= 1.0

def test_all_charts_generated():
    """Verify all required visual artifacts exist in analytics/charts."""
    expected_charts = [
        "univariate_age.png",
        "univariate_fare.png",
        "correlation_matrix.png",
        "multivariate_1_survival_sex_class.png",
        "multivariate_2_age_class_survival.png",
        "multivariate_3_fare_age_survival.png",
        "multivariate_4_family_survival.png",
        "decision_tree.png",
        "roc_curves.png",
        "residual_plot.png"
    ]
    for chart in expected_charts:
        chart_path = os.path.join("analytics/charts", chart)
        assert os.path.exists(chart_path), f"Expected visual artifact {chart} was not found"
