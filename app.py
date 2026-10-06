from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_DIR / "models"
# ============================================================
# LOAD SAVED MODELS
# ============================================================

regression_preprocessor = joblib.load(
    MODELS_DIR / "regression_preprocessor.pkl"
)

bulk_model = joblib.load(
    MODELS_DIR / "bulk_modulus_GPa_rf.pkl"
)

shear_model = joblib.load(
    MODELS_DIR / "shear_modulus_GPa_rf.pkl"
)

poisson_model = joblib.load(
    MODELS_DIR / "poisson_ratio_rf.pkl"
)

band_gap_model = joblib.load(
    MODELS_DIR / "band_gap_eV_rf.pkl"
)

melting_model = joblib.load(
    MODELS_DIR / "melting_point_K_rf.pkl"
)

classification_preprocessor = joblib.load(
    MODELS_DIR / "classification_preprocessor.pkl"
)

stability_model = joblib.load(
    MODELS_DIR / "stability_rf.pkl"
)
# ============================================================
# STREAMLIT PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="MatRisk-AI",
    page_icon="🧪",
    layout="wide"
)


# ============================================================
# PAGE TITLE
# ============================================================

st.title("🧪 MatRisk-AI")
st.subheader("AI-Based Materials Property Prediction")

st.write(
    "Enter material properties below to predict "
    "mechanical, electronic, thermal, and stability properties."
)
# ============================================================
# MATERIAL INPUTS
# ============================================================

st.header("Material Information")

col1, col2 = st.columns(2)

with col1:

    n_elements = st.number_input(
        "Number of Elements",
        min_value=1,
        value=2,
        step=1
    )

    crystal_system = st.selectbox(
        "Crystal System",
        [
            "triclinic",
            "monoclinic",
            "orthorhombic",
            "tetragonal",
            "trigonal",
            "hexagonal",
            "cubic"
        ]
    )

    spacegroup_number = st.number_input(
        "Spacegroup Number",
        min_value=1,
        max_value=230,
        value=225,
        step=1
    )

    category = st.text_input(
        "Material Category",
        value="oxide"
    )

    formation_energy_per_atom_eV = st.number_input(
        "Formation Energy per Atom (eV)",
        value=-1.0,
        format="%.6f"
    )

    energy_above_hull_eV = st.number_input(
        "Energy Above Hull (eV)",
        min_value=0.0,
        value=0.05,
        format="%.6f"
    )

with col2:

    is_metal = st.selectbox(
        "Is Metal?",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    density_g_cm3 = st.number_input(
        "Density (g/cm³)",
        min_value=0.0,
        value=5.0,
        format="%.4f"
    )

    nsites = st.number_input(
        "Number of Sites",
        min_value=1,
        value=4,
        step=1
    )

    volume_A3 = st.number_input(
        "Volume (Å³)",
        min_value=0.0,
        value=100.0,
        format="%.4f"
    )
    # ============================================================
# PREPARE INPUT DATA
# ============================================================

if st.button("🔮 Predict Material Properties"):

    regression_input = pd.DataFrame({
        "n_elements": [n_elements],
        "crystal_system": [crystal_system],
        "spacegroup_number": [spacegroup_number],
        "category": [category],
        "formation_energy_per_atom_eV": [
            formation_energy_per_atom_eV
        ],
        "energy_above_hull_eV": [
            energy_above_hull_eV
        ],
        "is_metal": [is_metal],
        "density_g_cm3": [density_g_cm3],
        "nsites": [nsites],
        "volume_A3": [volume_A3]
    })

    classification_input = pd.DataFrame({
        "n_elements": [n_elements],
        "crystal_system": [crystal_system],
        "spacegroup_number": [spacegroup_number],
        "formation_energy_per_atom_eV": [
            formation_energy_per_atom_eV
        ],
        "is_metal": [is_metal],
        "density_g_cm3": [density_g_cm3],
        "nsites": [nsites],
        "volume_A3": [volume_A3]
    })
        # ========================================================
    # REGRESSION PREDICTIONS
    # ========================================================

    regression_processed = regression_preprocessor.transform(
        regression_input
    )

    bulk_prediction = bulk_model.predict(
        regression_processed
    )[0]

    shear_prediction = shear_model.predict(
        regression_processed
    )[0]

    poisson_prediction = poisson_model.predict(
        regression_processed
    )[0]

    band_gap_prediction = band_gap_model.predict(
        regression_processed
    )[0]

    melting_prediction = melting_model.predict(
        regression_processed
    )[0]


    # ========================================================
    # STABILITY PREDICTION
    # ========================================================

    classification_processed = (
        classification_preprocessor.transform(
            classification_input
        )
    )

    stability_prediction = stability_model.predict(
        classification_processed
    )[0]

    stability_probability = stability_model.predict_proba(
        classification_processed
    )[0][1]
        # ========================================================
    # DISPLAY RESULTS
    # ========================================================

    st.header("Prediction Results")

    # --------------------------------------------------------
    # MECHANICAL PROPERTIES
    # --------------------------------------------------------

    st.subheader("Mechanical Properties")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Bulk Modulus",
            f"{bulk_prediction:.2f} GPa"
        )

    with col2:
        st.metric(
            "Shear Modulus",
            f"{shear_prediction:.2f} GPa"
        )

    with col3:
        st.metric(
            "Poisson Ratio",
            f"{poisson_prediction:.4f}"
        )


    # --------------------------------------------------------
    # ELECTRONIC & THERMAL PROPERTIES
    # --------------------------------------------------------

    st.subheader("Electronic & Thermal Properties")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Band Gap",
            f"{band_gap_prediction:.2f} eV"
        )

    with col2:
        st.metric(
            "Melting Point",
            f"{melting_prediction:.2f} K"
        )


    # --------------------------------------------------------
    # STABILITY
    # --------------------------------------------------------

    st.subheader("Stability Prediction")

    stability_label = (
        "Stable" if stability_prediction == 1
        else "Unstable"
    )

    stability_percent = stability_probability * 100

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Predicted Stability",
            stability_label
        )

    with col2:
        st.metric(
            "Stability Probability",
            f"{stability_percent:.2f}%"
        )
# ============================================================
# PROJECT INFORMATION
# ============================================================

st.divider()

st.caption(
    "MatRisk-AI | Machine Learning-Based Materials Property Prediction"
)

st.caption(
    "Models: Random Forest | "
    "Regression targets: 5 | "
    "Stability classification: leakage-safe"
)