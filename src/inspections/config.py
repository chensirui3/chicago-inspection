"""Paths, pinned data sources and project-wide constants."""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = Path(os.environ.get("INSPECTIONS_DATA_DIR", ROOT / "data"))
RAW_DIR = DATA_DIR / "raw"  # the city's .Rds files, untouched
PROCESSED_DIR = DATA_DIR / "processed"  # Parquet files this project creates
EXTERNAL_DIR = DATA_DIR / "external"  # the city's repo clone and portal downloads
PORTAL_DIR = EXTERNAL_DIR / "portal"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
TABLES_DIR = REPORTS_DIR / "tables"
CACHE_DIR = REPORTS_DIR / "cache"  # git-ignored intermediate results
SITE_DIR = ROOT / "site"  # Extension C static dashboard (GitHub Pages)

# The city's repository, pinned to the commit this project was built against (v1.7.0 merge,
# 2019-04-13). DATA/ was regenerated in that release; see "Row set" in README section 1.
CITY_REPO_URL = "https://github.com/Chicago/food-inspections-evaluation.git"
CITY_REPO_COMMIT = "603d8dba4d2ea266aced60ac49bdd6ee96704d05"

# SHA-256 of every .Rds file in the city's DATA/ folder at CITY_REPO_COMMIT.
CITY_DATA_SHA256 = {
    "11_bus_license.Rds": "835a33f51a3d91a3c20d2dc48d58027d2ae6416f2dab9909fabc19d22ede4a1f",
    "12_crime.Rds": "9bcb7e21b6be2600c3af538a0cddcae96f2d058be6df97d8442a17c887377567",
    "13_food_inspections.Rds": "37068d67ee120fbd06c01b6bb87873dd613eb01ee2588c1c49087c1160bb09b6",
    "14_garbage_carts.Rds": "9ad4fbc52d6059b9f776da61fe78832ac7376650e556ecf0822c5f2ab8de4a17",
    "15_sanitation_code.Rds": "867c9ce9a1551c2390405317becd144e178c8031f0632b52d1aac02cfeaed1aa",
    "17_mongo_weather_update.Rds": (
        "cbca87afaab1c54a16d91bb87c8a82d1de6d2734406a46527b880441a64a7319"
    ),
    "19_inspector_assignments.Rds": (
        "8296367af49a4c67ef9a84a4d71e590c5727343b61dd6bcc85bfbbe2c7014d95"
    ),
    "21_food_inspection_violation_matrix.Rds": (
        "3add3cf38387f3b9c4764fb1b5a683fb1869b852b76264c7dbd3db4c9a4bed55"
    ),
    "21_food_inspection_violation_matrix_nums.Rds": (
        "09506f1d21b87a416befef0c1ef01e79dd62bb23f1b5ee3a7c6bf3e26395c3c9"
    ),
    "22_burglary_heat.Rds": "f72119f0ec8d72503d381c090bd31214272af07c931e351ff43abce70df84f44",
    "22_garbageCarts_heat.Rds": "5f8a8663c6969c70d67b03c23dd5c0af8a90bc718b58ad299394350548a72695",
    "22_sanitationComplaints_heat.Rds": (
        "d8d12612814425beee31aafe685e9e1eef51c3c5066b0f7c90a81ee272ed7d9e"
    ),
    "23_food_insp_features.Rds": "b4c3ed9451087d947f8cb323fb40f79b364e74affde5d4d8bd64d5f36adf3d43",
    "24_bus_features.Rds": "277948a125f59f598aeaf512f074b5bdda390131238db85c4cfa04e5ef787ffc",
    "30_xgboost_data.Rds": "e8ab3f1ec2b4523805c084b7ec1e561c3ff213a9f9c8664fe6ca26525081cde2",
    "30_xgboost_model.Rds": "3070d5a0b838fb6ad9ab3424ad67e46cdb1b1d498aea81f5b2b581cc92d174ba",
    "inspectors.Rds": "1aec5e49da83b2d0e5a8626b1c80fdf8d3f1ce217c2dc0c2186c08640aef48cf",
}

SEED = 2014  # every random draw in the project derives from this seed
N_BOOT = 1000  # bootstrap replicates for every interval in the README

# The six anonymized inspector clusters, in the city's column order.
INSPECTOR_CLUSTERS = ("blue", "brown", "green", "orange", "purple", "yellow")

# The ten non-inspector variables of the city's model, in its column order (CODE/30a).
CITY_FEATURES = (
    "pastSerious",
    "pastCritical",
    "timeSinceLast",
    "ageAtInspection",
    "consumption_on_premises_incidental_activity",
    "tobacco",
    "temperatureMax",
    "heat_burglary",
    "heat_sanitation",
    "heat_garbage",
)

# Published benchmark (City of Chicago technical report and code comments).
PUBLISHED = {
    "days_earlier": 7.44,
    "days_earlier_sd": 25.20,
    "n_test": 1637,
    "n_positive": 258,
    "n_train": 17075,
    "n_later": 99,
    "first_half_actual": 0.5465116,
    "first_half_model": 0.6821705,
    "xgboost_days_earlier": 7.79,
    "xgboost_auc": 0.6790282,
    "random_forest_days_earlier": 6.55,
}

# Coefficients in the report's "Table of Coefficients" (glmnet, 100th lambda, training data).
PUBLISHED_COEFFICIENTS = {
    "Inspector_blue": 0.950,
    "Inspector_brown": -1.306,
    "Inspector_green": -0.244,
    "Inspector_orange": 0.202,
    "Inspector_purple": 1.555,
    "Inspector_yellow": -0.697,
    "pastSerious": 0.302,
    "pastCritical": 0.427,
    "timeSinceLast": 0.097,
    "ageAtInspection": -0.164,
    "consumption_on_premises_incidental_activity": 0.411,
    "tobacco": 0.171,
    "temperatureMax": 0.005,
    "heat_burglary": 0.002,
    "heat_sanitation": 0.002,
    "heat_garbage": -0.004,
}

# Chicago's new food code took effect on this date (critical/serious -> Priority/PF).
FOOD_CODE_CHANGE = "2018-07-01"
