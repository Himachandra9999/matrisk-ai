from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

DATA_DIR = PROJECT_DIR / "data" / "raw"
MODELS_DIR = PROJECT_DIR / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# DATA FILE
# ============================================================

DS1_PATH = DATA_DIR / "DS1_material_properties_5500.csv"

print("=" * 70)
print("MatRisk-AI Model Training")
print("=" * 70)

print("\nProject directory:")
print(PROJECT_DIR)

print("\nDataset:")
print(DS1_PATH)
# ============================================================
# LOAD DATASET
# ============================================================

df_material = pd.read_csv(DS1_PATH)

print("\nDS1 loaded successfully!")
print("Shape:", df_material.shape)
print("Columns:")
print(df_material.columns.tolist())
# ============================================================
# CREATE MATERIAL RECORD ID
# ============================================================

df_material["material_record_id"] = (
    df_material["material_id"].astype(str)
    + "_"
    + df_material["formula"].astype(str)
)

# ============================================================
# CREATE FINAL DATASET
# ============================================================

df_final = df_material.copy()

# ============================================================
# PHYSICS VALIDATION
# ============================================================

df_final["calculated_poisson_ratio"] = (
    (3 * df_final["bulk_modulus_GPa"]
     - 2 * df_final["shear_modulus_GPa"])
    /
    (2 * (
        3 * df_final["bulk_modulus_GPa"]
        + df_final["shear_modulus_GPa"]
    ))
)

df_final["physics_error"] = (
    df_final["poisson_ratio"]
    - df_final["calculated_poisson_ratio"]
).abs()

# ============================================================
# CHECK FINAL DATASET
# ============================================================

print("\nFinal dataset created!")
print("Shape:", df_final.shape)

print("\nFinal columns:")
print(df_final.columns.tolist())

print("\nPhysics error statistics:")
print(df_final["physics_error"].describe())
# ============================================================
# REGRESSION FEATURES
# ============================================================

ml_features = [
    "n_elements",
    "crystal_system",
    "spacegroup_number",
    "category",
    "formation_energy_per_atom_eV",
    "energy_above_hull_eV",
    "is_metal",
    "density_g_cm3",
    "nsites",
    "volume_A3"
]

X = df_final[ml_features].copy()

# Regression targets
y_bulk = df_final["bulk_modulus_GPa"].copy()
y_shear = df_final["shear_modulus_GPa"].copy()
y_poisson = df_final["poisson_ratio"].copy()
y_band_gap = df_final["band_gap_eV"].copy()
y_melting = df_final["melting_point_K"].copy()

print("\nRegression features created.")
print("X shape:", X.shape)

print("\nRegression targets:")
print("Bulk modulus :", y_bulk.shape)
print("Shear modulus:", y_shear.shape)
print("Poisson ratio:", y_poisson.shape)
print("Band gap     :", y_band_gap.shape)
print("Melting point:", y_melting.shape)
# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

indices = df_final.index

train_indices, test_indices = train_test_split(
    indices,
    test_size=0.20,
    random_state=42,
    stratify=df_final["is_stable"]
)

# Create X train/test
X_train = X.loc[train_indices].copy()
X_test = X.loc[test_indices].copy()

# Create target train/test sets
y_bulk_train = y_bulk.loc[train_indices]
y_bulk_test = y_bulk.loc[test_indices]

y_shear_train = y_shear.loc[train_indices]
y_shear_test = y_shear.loc[test_indices]

y_poisson_train = y_poisson.loc[train_indices]
y_poisson_test = y_poisson.loc[test_indices]

y_band_gap_train = y_band_gap.loc[train_indices]
y_band_gap_test = y_band_gap.loc[test_indices]

y_melting_train = y_melting.loc[train_indices]
y_melting_test = y_melting.loc[test_indices]

y_stable_train = df_final.loc[train_indices, "is_stable"]
y_stable_test = df_final.loc[test_indices, "is_stable"]

print("\nTrain/test split completed.")
print("X_train shape:", X_train.shape)
print("X_test shape :", X_test.shape)

print("\nStability distribution:")
print("Training:")
print(y_stable_train.value_counts())

print("\nTesting:")
print(y_stable_test.value_counts())
# ============================================================
# REGRESSION PREPROCESSING
# ============================================================

numerical_features = [
    "n_elements",
    "spacegroup_number",
    "formation_energy_per_atom_eV",
    "energy_above_hull_eV",
    "is_metal",
    "density_g_cm3",
    "nsites",
    "volume_A3"
]

categorical_features = [
    "crystal_system",
    "category"
]

numerical_transformer = StandardScaler()

categorical_transformer = OneHotEncoder(
    handle_unknown="ignore"
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numerical_transformer,
            numerical_features
        ),
        (
            "cat",
            categorical_transformer,
            categorical_features
        )
    ]
)

# Fit only on training data
X_train_processed = preprocessor.fit_transform(X_train)

# Transform test data
X_test_processed = preprocessor.transform(X_test)

print("\nRegression preprocessing completed.")
print("X_train_processed shape:", X_train_processed.shape)
print("X_test_processed shape :", X_test_processed.shape)
# ============================================================
# TRAIN RANDOM FOREST REGRESSION MODELS
# ============================================================

regression_targets = {
    "bulk_modulus_GPa": y_bulk_train,
    "shear_modulus_GPa": y_shear_train,
    "poisson_ratio": y_poisson_train,
    "band_gap_eV": y_band_gap_train,
    "melting_point_K": y_melting_train
}

rf_models = {}

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST REGRESSION MODELS")
print("=" * 70)

for target, y_train_target in regression_targets.items():

    print(f"\nTraining model for: {target}")

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train_processed,
        y_train_target
    )

    rf_models[target] = model

    print("Completed.")
    # ============================================================
# SAVE REGRESSION MODELS
# ============================================================

print("\n" + "=" * 70)
print("SAVING REGRESSION MODELS")
print("=" * 70)

# Save the regression preprocessor
joblib.dump(
    preprocessor,
    MODELS_DIR / "regression_preprocessor.pkl"
)

print("\nSaved:")
print("regression_preprocessor.pkl")

# Save each regression model
for target, model in rf_models.items():

    model_path = MODELS_DIR / f"{target}_rf.pkl"

    joblib.dump(
        model,
        model_path
    )

    print(model_path.name)

print("\nAll regression models saved successfully!")
# ============================================================
# LEAKAGE-SAFE STABILITY CLASSIFICATION
# ============================================================

classification_features = [
    "n_elements",
    "crystal_system",
    "spacegroup_number",
    "formation_energy_per_atom_eV",
    "is_metal",
    "density_g_cm3",
    "nsites",
    "volume_A3"
]

X_classification = df_final[classification_features].copy()
y_classification = df_final["is_stable"].copy()

# Use the same train/test indices as regression
X_class_train = X_classification.loc[train_indices].copy()
X_class_test = X_classification.loc[test_indices].copy()

y_class_train = y_classification.loc[train_indices]
y_class_test = y_classification.loc[test_indices]

# Classification preprocessing
classification_numerical_features = [
    "n_elements",
    "spacegroup_number",
    "formation_energy_per_atom_eV",
    "is_metal",
    "density_g_cm3",
    "nsites",
    "volume_A3"
]

classification_categorical_features = [
    "crystal_system"
]

classification_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            StandardScaler(),
            classification_numerical_features
        ),
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            classification_categorical_features
        )
    ]
)

# Fit only on training data
X_class_train_processed = (
    classification_preprocessor.fit_transform(X_class_train)
)

# Transform test data
X_class_test_processed = (
    classification_preprocessor.transform(X_class_test)
)

print("\n" + "=" * 70)
print("LEAKAGE-SAFE STABILITY CLASSIFICATION")
print("=" * 70)

print("\nClassification features:")
print(classification_features)

print("\nX_class_train shape:",
      X_class_train.shape)

print("X_class_test shape :",
      X_class_test.shape)

print("\nProcessed shapes:")
print("X_class_train_processed shape:",
      X_class_train_processed.shape)

print("X_class_test_processed shape :",
      X_class_test_processed.shape)
# ============================================================
# TRAIN RANDOM FOREST STABILITY CLASSIFIER
# ============================================================

print("\n" + "=" * 70)
print("TRAINING RANDOM FOREST STABILITY CLASSIFIER")
print("=" * 70)

rf_classifier = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    n_jobs=-1
)

rf_classifier.fit(
    X_class_train_processed,
    y_class_train
)

print("\nRandom Forest stability classifier trained successfully!")
# ============================================================
# SAVE STABILITY CLASSIFIER
# ============================================================

print("\n" + "=" * 70)
print("SAVING STABILITY CLASSIFIER")
print("=" * 70)

# Save classification preprocessor
joblib.dump(
    classification_preprocessor,
    MODELS_DIR / "classification_preprocessor.pkl"
)

print("\nSaved:")
print("classification_preprocessor.pkl")

# Save stability classifier
joblib.dump(
    rf_classifier,
    MODELS_DIR / "stability_rf.pkl"
)

print("stability_rf.pkl")

print("\nStability classifier saved successfully!")
# ============================================================
# VERIFY SAVED MODEL FILES
# ============================================================

print("\n" + "=" * 70)
print("VERIFYING SAVED MODEL FILES")
print("=" * 70)

model_files = [
    "regression_preprocessor.pkl",
    "bulk_modulus_GPa_rf.pkl",
    "shear_modulus_GPa_rf.pkl",
    "poisson_ratio_rf.pkl",
    "band_gap_eV_rf.pkl",
    "melting_point_K_rf.pkl",
    "classification_preprocessor.pkl",
    "stability_rf.pkl"
]

all_files_exist = True

for filename in model_files:

    file_path = MODELS_DIR / filename

    if file_path.exists():
        print(f"✓ {filename}")
    else:
        print(f"✗ {filename}")
        all_files_exist = False

print("\n" + "=" * 70)

if all_files_exist:
    print("ALL MODEL FILES VERIFIED SUCCESSFULLY!")
else:
    print("WARNING: SOME MODEL FILES ARE MISSING.")

print("=" * 70)