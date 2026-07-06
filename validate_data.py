
import pandas as pd
import numpy as np

# Load cleaned dataset
df = pd.read_csv("cleaned_credit_default.csv")

print("Automated Data Validation Started")
print("--------------------------------")

# Check 1: Missing values
total_missing = df.isnull().sum().sum()
print("Total missing values:", total_missing)

# Check 2: Duplicate rows
duplicate_rows = df.duplicated().sum()
print("Duplicate rows:", duplicate_rows)

# Check 3: Duplicate customer IDs
duplicate_ids = df["ID"].duplicated().sum()
print("Duplicate IDs:", duplicate_ids)

# Check 4: Target column exists
target_exists = "DEFAULT_NEXT_MONTH" in df.columns
print("Target column exists:", target_exists)

# Check 5: Target values are valid
valid_target_values = set(df["DEFAULT_NEXT_MONTH"].unique()).issubset({0, 1})
print("Target values valid:", valid_target_values)

# Check 6: EDUCATION values are valid after cleaning
valid_education_values = set(df["EDUCATION"].unique()).issubset({1, 2, 3, 4})
print("EDUCATION values valid:", valid_education_values)

# Check 7: MARRIAGE values are valid after cleaning
valid_marriage_values = set(df["MARRIAGE"].unique()).issubset({1, 2, 3})
print("MARRIAGE values valid:", valid_marriage_values)

# Check 8: Infinite values
total_infinite = np.isinf(df.select_dtypes(include=[np.number])).sum().sum()
print("Total infinite values:", total_infinite)

print("--------------------------------")

# Final validation result
if (
    total_missing == 0
    and duplicate_rows == 0
    and duplicate_ids == 0
    and target_exists
    and valid_target_values
    and valid_education_values
    and valid_marriage_values
    and total_infinite == 0
):
    print("VALIDATION PASSED: Dataset is ready for modelling and dashboard development.")
else:
    print("VALIDATION FAILED: Dataset requires further cleaning.")
