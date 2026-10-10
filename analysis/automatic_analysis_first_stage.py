# TEST
import pandas as pd
import numpy as np
import os 
import scipy
import json
import anthropic
import hashlib

REPORT_SCHEMA_GLOSSARY = """
Field definitions for the EDA summary below:

- data_profile.shape_report: row and column counts of the dataset.
- data_profile.dtype_report: which columns are numeric vs non-numeric.
- missingness.missing_percentage: proportion (0-1 scale) of missing values per column.
- missingness.alarming_cols: columns whose missingness exceeds a concerning threshold.
- missingness.suspected_missing_at_random_numerical/categorical: columns whose missingness
  significantly correlates with another observed column (evidence of MAR), with p-values from
  point-biserial correlation (numeric) or chi-square test (categorical). Does NOT prove MAR over
  MNAR — only indicates missingness is not purely random with respect to the named column.
- missingness.perc_invalid_dates: proportion of unparseable date values, for datetime columns.
- target_leakage.detected_numerical/categorical_leakages: features whose correlation with the
  target exceeds 0.9 (Pearson/point-biserial for numeric, Cramér's V for categorical) alongside
  statistical significance (p<0.05). High correlation may indicate a derived/redundant feature,
  OR a genuinely strong, legitimate predictor — domain judgment decides; do not assume a flagged
  feature must be dropped.
- outliers.<column>: proportion of values below/above/either side of the IQR fences
  (Q1 - 1.5*IQR, Q3 + 1.5*IQR). Can miss physically implausible values if the distribution is
  wide, and can over-flag on heavily skewed distributions.
- numeric_column_distributions.<column>: mean, std, min/max, quartiles, skew, kurtosis.
  Skew > ~1 and kurtosis > ~3 indicate meaningful departure from normality.
- target_profile.class_imbalance: proportion of rows per target class, or target distribution
  stats for regression.
"""

TARGET_COL = "lv_active_power"
BUSINESS_CONTEXT = "wind turbine power output prediction from sensor data"
CACHE_DIR = "advice_cache"

client = anthropic.Anthropic()

def build_prompt(report: dict) -> str:
    return f"""You are advising on a baseline modelling approach.

{REPORT_SCHEMA_GLOSSARY}

Here is a structured EDA summary for this dataset: {json.dumps(report, indent=2)}
The prediction target is: {TARGET_COL}
The business context is: {BUSINESS_CONTEXT}

Suggest:
1. A baseline model family and why
2. Key risks flagged by the EDA that the approach should account for
3. What NOT to do given the flagged issues (e.g. leakage columns to drop)
"""


def get_cache_key(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()


def get_or_generate_advice(prompt: str) -> str:
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{get_cache_key(prompt)}.txt")

    if os.path.exists(cache_path):
        print("Using cached advice — no API call made")
        with open(cache_path, "r") as f:
            return f.read()

    response = client.messages.create(
    model="claude-sonnet-5",
    max_tokens=1000,
    thinking={"type": "disabled"},
    messages=[{"role": "user", "content": prompt}],
    )
    advice = response.content[0].text

    with open(cache_path, "w") as f:
        f.write(advice)

    return advice


def main():
    with open("../analysis/eda_report.json", "r") as f:
        
      report = json.load(f)

    prompt = build_prompt(report)
    advice = get_or_generate_advice(prompt)

    with open("advice_report.md", "w") as f:
        f.write(f"# EDA Advisor Report\n\n{advice}\n")

    print("Report saved to advice_report.md")


if __name__ == "__main__":
    main()