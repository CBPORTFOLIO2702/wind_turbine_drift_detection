
import os 
import json
import anthropic
import hashlib

REPORT_SCHEMA_GLOSSARY = """
Field definitions for the model evaluation summary below. Field names under "features" and
"hyperparams" are project-specific and vary by run — their keys are the actual column/parameter
names used in this project, and their values are plain-language descriptions or config values
supplied at report-generation time, not part of a fixed schema.

- splitting_strategy: how the dataset was divided for training/validation/testing.
  - name: the splitting method used (e.g. random split, time-based split, stratified split).
    If chronological ("time series split" or similar), no row used for validation or test was
    seen during training, and no preprocessing statistic (imputation values, scaling, etc.)
    should have been computed using rows from validation or test.
  - any threshold/boundary fields (e.g. train_threshold, val_threshold): the cutoffs defining
    each split, in whatever unit is relevant to the splitting method (timestamp for time-based,
    row index or fraction for random).

- features: a dictionary of every feature fed to the model, keyed by its column name, with a
  plain-language description of what that feature represents as the value. Use these
  descriptions to understand what each feature physically or logically means — do not assume
  meaning from the column name alone, as naming conventions vary by project.

- model: the algorithm family used for this run (e.g. a specific regressor or classifier class
  name). Interpret performance and feature-importance expectations in light of this model type
  (e.g. tree-based models are not sensitive to feature scaling or monotonic transforms; linear
  models are).

- hyperparams: a dictionary of the exact configuration used to train this model instance, keyed
  by parameter name. Parameters affecting model capacity/complexity (e.g. depth, number of
  estimators, regularization strength) are relevant to assessing overfitting risk; parameters
  like random seeds or parallelism settings are not relevant to model quality.

- metrics: performance of the trained model, evaluated on held-out data (validation and/or test,
  depending on what's included). Field names depend on the task type:
  - Regression tasks typically report mae (Mean Absolute Error), rmse (Root Mean Squared Error),
    and r2 (R-squared). A large gap between mae and rmse indicates a subset of predictions with
    unusually large errors rather than uniformly distributed error. r2 indicates proportion of
    variance explained but says nothing about where or why the model is wrong.
  - Classification tasks typically report accuracy, precision, recall, f1, and/or AUC. Accuracy
    alone can be misleading under class imbalance — check for a class_imbalance or similar field
    in the underlying EDA report before treating accuracy as sufficient.
  - If metrics are reported per split (e.g. train vs validation vs test), a large gap between
    train and held-out performance indicates overfitting.
"""

TARGET_COL = "lv_active_power"
BUSINESS_CONTEXT = "wind turbine power output prediction from sensor data, to predict expected output power. The goal will be to compare this to the design output power, which is a column within the dataset but not to be used as a feature."
CACHE_DIR = "advice_cache"

client = anthropic.Anthropic()

def build_prompt(report: dict) -> str:
    return f"""You are advising on a baseline modelling approach.

{REPORT_SCHEMA_GLOSSARY}

Here is a structured EDA summary for this dataset: {json.dumps(report, indent=2)}
The prediction target is: {TARGET_COL}
The business context is: {BUSINESS_CONTEXT}

Write:
1. A written description what the model results show
2. Next steps for model improvements
3. A list of 3-5 feature engineering ideas to improve the model
4. A list of 3-5 hyperparameter tuning ideas to improve the model

Constraints:
- Keep as concise as possible so that the report does not get truncated.
- Ensure that advice is prioritised. This report is meant to help someone make thier model better.
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
    with open("analysis/gold/eda_report.json", "r") as f:
        
      report = json.load(f)

    prompt = build_prompt(report)
    advice = get_or_generate_advice(prompt)

    with open("advice_report.md", "w") as f:
        f.write(f"# EDA Advisor Report\n\n{advice}\n")

    print("Report saved to advice_report.md")


if __name__ == "__main__":
    main()