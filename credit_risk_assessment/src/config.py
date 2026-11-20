"""Project configuration, schema definitions, and domain cost constants."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT.parent / "statlog+german+credit+data"
GERMAN_DATA_PATH = RAW_DATA_DIR / "german.data"
GERMAN_NUMERIC_PATH = RAW_DATA_DIR / "german.data-numeric"
GERMAN_DOC_PATH = RAW_DATA_DIR / "german.doc"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

RANDOM_STATE = 42
TEST_SIZE = 0.20
N_SPLITS = 5

# Cost matrix from german.doc (Section 8):
# Positive class (1) = Bad risk (raw target == 2)
# Negative class (0) = Good risk (raw target == 1)
# Predicting Good (0) when actual is Bad (1) => False Negative costs 5
# Predicting Bad (1) when actual is Good (0) => False Positive costs 1
COST_FN = 5.0
COST_FP = 1.0
BAYES_OPTIMAL_THRESHOLD = COST_FP / (COST_FP + COST_FN)  # 1/6 ~= 0.1667

COLUMN_NAMES = [
    "checking_status",          # Attr 1 (qualitative)
    "duration",                 # Attr 2 (numerical - months)
    "credit_history",           # Attr 3 (qualitative)
    "purpose",                  # Attr 4 (qualitative)
    "credit_amount",            # Attr 5 (numerical - DM)
    "savings_status",           # Attr 6 (qualitative)
    "employment",               # Attr 7 (qualitative)
    "installment_rate",         # Attr 8 (discrete numerical 1..4)
    "personal_status_sex",      # Attr 9 (qualitative)
    "other_debtors",            # Attr 10 (qualitative)
    "residence_since",          # Attr 11 (discrete numerical 1..4)
    "property_magnitude",       # Attr 12 (qualitative)
    "age",                      # Attr 13 (numerical - years)
    "other_installment_plans",  # Attr 14 (qualitative)
    "housing",                  # Attr 15 (qualitative)
    "existing_credits",         # Attr 16 (discrete numerical 1..4)
    "job",                      # Attr 17 (qualitative)
    "people_liable",            # Attr 18 (discrete numerical 1..2)
    "telephone",                # Attr 19 (qualitative)
    "foreign_worker",           # Attr 20 (qualitative)
    "raw_target",               # 1 = Good, 2 = Bad
]

CONTINUOUS_COLS = ["duration", "credit_amount", "age"]
DISCRETE_NUM_COLS = [
    "installment_rate",
    "residence_since",
    "existing_credits",
    "people_liable",
]
CATEGORICAL_COLS = [
    "checking_status",
    "credit_history",
    "purpose",
    "savings_status",
    "employment",
    "personal_status_sex",
    "other_debtors",
    "property_magnitude",
    "other_installment_plans",
    "housing",
    "job",
    "telephone",
    "foreign_worker",
]

# Full codebook documented in german.doc (includes dead codes A47 and A95 for audit)
DOC_CODEBOOK = {
    "checking_status": ["A11", "A12", "A13", "A14"],
    "credit_history": ["A30", "A31", "A32", "A33", "A34"],
    "purpose": ["A40", "A41", "A42", "A43", "A44", "A45", "A46", "A47", "A48", "A49", "A410"],
    "savings_status": ["A61", "A62", "A63", "A64", "A65"],
    "employment": ["A71", "A72", "A73", "A74", "A75"],
    "personal_status_sex": ["A91", "A92", "A93", "A94", "A95"],
    "other_debtors": ["A101", "A102", "A103"],
    "property_magnitude": ["A121", "A122", "A123", "A124"],
    "other_installment_plans": ["A141", "A142", "A143"],
    "housing": ["A151", "A152", "A153"],
    "job": ["A171", "A172", "A173", "A174"],
    "telephone": ["A191", "A192"],
    "foreign_worker": ["A201", "A202"],
}

# Human-readable labels for interpretable Decision Trees and Feature Importance charts
READABLE_MAPS = {
    "checking_status": {
        "A11": "lt_0_DM",
        "A12": "0_to_200_DM",
        "A13": "ge_200_DM",
        "A14": "no_checking",
    },
    "credit_history": {
        "A30": "no_credits_or_all_paid",
        "A31": "all_paid_this_bank",
        "A32": "existing_paid_duly",
        "A33": "past_delay",
        "A34": "critical_or_other_bank",
    },
    "purpose": {
        "A40": "car_new",
        "A41": "car_used",
        "A42": "furniture_equip",
        "A43": "radio_tv",
        "A44": "domestic_app",
        "A45": "repairs",
        "A46": "education",
        "A48": "retraining",
        "A49": "business",
        "A410": "others",
    },
    "savings_status": {
        "A61": "lt_100_DM",
        "A62": "100_to_500_DM",
        "A63": "500_to_1000_DM",
        "A64": "ge_1000_DM",
        "A65": "unknown_or_no_savings",
    },
    "employment": {
        "A71": "unemployed",
        "A72": "lt_1yr",
        "A73": "1_to_4yr",
        "A74": "4_to_7yr",
        "A75": "ge_7yr",
    },
    "personal_status_sex": {
        "A91": "male_div_sep",
        "A92": "female_div_sep_mar",
        "A93": "male_single",
        "A94": "male_mar_wid",
    },
    "other_debtors": {
        "A101": "none",
        "A102": "co_applicant",
        "A103": "guarantor",
    },
    "property_magnitude": {
        "A121": "real_estate",
        "A122": "life_ins_or_society",
        "A123": "car_or_other",
        "A124": "unknown_or_no_prop",
    },
    "other_installment_plans": {
        "A141": "bank",
        "A142": "stores",
        "A143": "none",
    },
    "housing": {
        "A151": "rent",
        "A152": "own",
        "A153": "for_free",
    },
    "job": {
        "A171": "unemp_unskilled_nonres",
        "A172": "unskilled_res",
        "A173": "skilled",
        "A174": "mgmt_selfemp_high",
    },
    "telephone": {
        "A191": "none",
        "A192": "registered",
    },
    "foreign_worker": {
        "A201": "yes",
        "A202": "no",
    },
}

RARE_PURPOSE_CODES = {"A44", "A45", "A48", "A410"}  # Each < 3% of dataset (55 rows combined)

EMPLOYMENT_ORDER = {"A71": 0, "A72": 1, "A73": 2, "A74": 3, "A75": 4}
JOB_ORDER = {"A171": 0, "A172": 1, "A173": 2, "A174": 3}

AGE_DOMAIN_EDGES = [18, 25, 35, 50, 100]
AGE_DOMAIN_LABELS = ["19_25", "26_35", "36_50", "51_75"]

