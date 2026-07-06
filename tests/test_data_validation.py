import pandas as pd
import numpy as np
from pathlib import Path

DATA_PATH = Path("cleaned_credit_default.csv")

def test_dataset_exists():
    assert DATA_PATH.exists(), "Cleaned dataset file is missing."

def test_no_missing_values():
    df = pd.read_csv(DATA_PATH)
    assert df.isnull().sum().sum() == 0, "Dataset contains missing values."

def test_no_duplicate_ids():
    df = pd.read_csv(DATA_PATH)
    assert df["ID"].duplicated().sum() == 0, "Dataset contains duplicate customer IDs."

def test_target_column_exists():
    df = pd.read_csv(DATA_PATH)
    assert "DEFAULT_NEXT_MONTH" in df.columns, "Target column is missing."

def test_target_values_are_valid():
    df = pd.read_csv(DATA_PATH)
    valid_values = set(df["DEFAULT_NEXT_MONTH"].unique())
    assert valid_values.issubset({0, 1}), "Target column contains invalid values."

def test_education_values_are_valid():
    df = pd.read_csv(DATA_PATH)
    valid_values = set(df["EDUCATION"].unique())
    assert valid_values.issubset({1, 2, 3, 4}), "EDUCATION column contains invalid values."

def test_marriage_values_are_valid():
    df = pd.read_csv(DATA_PATH)
    valid_values = set(df["MARRIAGE"].unique())
    assert valid_values.issubset({1, 2, 3}), "MARRIAGE column contains invalid values."
