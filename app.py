from __future__ import annotations

import os
import site
import sys
from pathlib import Path
from typing import Any


APP_DIR = Path(__file__).resolve().parent
PCC_DIR = APP_DIR / "pcc"


def configure_runtime_paths() -> None:
    """Use clean dependencies locally while leaving cloud virtualenvs alone."""
    if os.name == "nt":
        os.environ["PYTHONNOUSERSITE"] = "1"
        user_site = site.getusersitepackages()
        sys.path = [
            item
            for item in sys.path
            if item != user_site and "AppData\\Roaming\\Python\\Python313\\site-packages" not in item
        ]

    for parent in (APP_DIR, *APP_DIR.parents):
        local_deps = parent / "_appdeps2"
        if local_deps.is_dir() and str(local_deps) not in sys.path:
            sys.path.append(str(local_deps))
            break


configure_runtime_paths()

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import streamlit as st


MODEL_SPECS = {
    "complication": {
        "title": "Complication Model",
        "model_path": PCC_DIR / "rf_pcc_stage1_complication_model.joblib",
        "cutoff": 0.215672,
    },
    "plos_conditional": {
        "title": "PCC PLOS Model",
        "model_path": PCC_DIR / "rf_pcc_stage2_plos_given_complication_model.joblib",
        "cutoff": 0.295133,
    },
}

FEATURES = [
    {
        "key": "pni",
        "label": "PNI",
        "unit": "",
        "default": 51.50,
        "min": 0.0,
        "max": 100.0,
        "step": 0.1,
        "format": "%.2f",
        "help": "PNI = albumin + 5 x total lymphocyte count",
    },
    {
        "key": "bmi",
        "label": "BMI",
        "unit": "kg/m^2",
        "default": 22.49,
        "min": 10.0,
        "max": 60.0,
        "step": 0.1,
        "format": "%.2f",
        "help": None,
    },
    {
        "key": "age",
        "label": "Age",
        "unit": "years",
        "default": 65.0,
        "min": 0.0,
        "max": 120.0,
        "step": 1.0,
        "format": "%.0f",
        "help": None,
    },
    {
        "key": "bloodloss",
        "label": "Intraoperative Blood Loss",
        "unit": "mL",
        "default": 100.0,
        "min": 0.0,
        "max": 5000.0,
        "step": 10.0,
        "format": "%.0f",
        "help": None,
    },
    {
        "key": "optime",
        "label": "Operation Time",
        "unit": "min",
        "default": 270.0,
        "min": 0.0,
        "max": 1500.0,
        "step": 5.0,
        "format": "%.0f",
        "help": None,
    },
    {
        "key": "tumorsize",
        "label": "Maximum Tumor Diameter",
        "unit": "cm",
        "default": 3.50,
        "min": 0.0,
        "max": 60.0,
        "step": 0.1,
        "format": "%.2f",
        "help": None,
    },
    {
        "key": "cea",
        "label": "CEA",
        "unit": "ng/mL",
        "default": 2.89,
        "min": 0.0,
        "max": 1000.0,
        "step": 0.1,
        "format": "%.2f",
        "help": None,
    },
]


st.set_page_config(
    page_title="PCC Postoperative Risk Prediction Model",
    page_icon="",
    layout="centered",
    initial_sidebar_state="collapsed",
)


def inject_css() -> None:
    st.html(
        """
        <style>
            :root {
                --ink: #13254a;
                --muted: #657184;
                --line: #e2e8f0;
                --blue: #2f6fe4;
                --card: rgba(255, 255, 255, 0.94);
                --danger: #f04452;
            }

            .stApp {
                background: linear-gradient(180deg, #f7fbff 0%, #eef4fb 100%);
                color: var(--ink);
            }

            .block-container {
                max-width: 1120px;
                padding-top: 4.2rem;
                padding-left: 1.4rem;
                padding-right: 1.4rem;
                padding-bottom: 2.5rem;
            }

            h1, h2, h3, p {
                letter-spacing: 0;
            }

            .hero {
                text-align: center;
                margin-bottom: 1.6rem;
            }

            .hero h1 {
                color: var(--ink);
                font-size: clamp(2.2rem, 4vw, 2.9rem);
                line-height: 1.2;
                margin: 0 0 0.65rem;
                font-weight: 800;
            }

            .hero p {
                color: #5b6577;
                font-size: 1.16rem;
                margin: 0;
            }

            [data-testid="stForm"],
            .result-panel,
            .note-panel {
                background: var(--card);
                border: 1px solid rgba(201, 211, 224, 0.86);
                border-radius: 8px;
                box-shadow: 0 10px 28px rgba(17, 39, 76, 0.10);
                padding: 1.55rem 1.85rem 1.45rem;
            }

            [data-testid="stForm"] {
                margin-bottom: 1.45rem;
            }

            .section-title {
                display: flex;
                align-items: center;
                gap: 0.7rem;
                color: #165cc8;
                font-size: 1.45rem;
                font-weight: 780;
                margin: 0 0 1.2rem;
                padding-bottom: 1.15rem;
                border-bottom: 1px solid var(--line);
            }

            .title-icon {
                width: 1.05rem;
                height: 1.35rem;
                border-radius: 4px;
                background: linear-gradient(180deg, #2f78ec, #1d57be);
                box-shadow: inset 0.28rem 0 0 rgba(255, 255, 255, 0.42);
                display: inline-block;
            }

            .input-label {
                min-height: 2.75rem;
                display: flex;
                align-items: center;
                font-size: 1.02rem;
                color: #17223f;
                font-weight: 620;
            }

            .unit-label {
                min-height: 2.75rem;
                display: flex;
                align-items: center;
                color: #667085;
                font-size: 0.98rem;
                white-space: nowrap;
            }

            .row-divider {
                height: 1px;
                background: var(--line);
                margin: 0.3rem 0 0.7rem;
            }

            div[data-testid="stNumberInput"] input {
                min-height: 2.65rem;
                border-radius: 7px;
                border-color: #cfd8e5;
                color: #17223f;
                font-size: 1rem;
            }

            div[data-testid="stNumberInput"] input:focus {
                border-color: var(--blue);
                box-shadow: 0 0 0 1px rgba(47, 111, 228, 0.18);
            }

            div[data-testid="stFormSubmitButton"] button {
                height: 3.15rem;
                border-radius: 7px;
                background: linear-gradient(180deg, #3478ee 0%, #2363d7 100%);
                color: white;
                border: 0;
                font-size: 1.08rem;
                font-weight: 760;
                margin-top: 0.65rem;
            }

            div[data-testid="stFormSubmitButton"] button:hover {
                background: linear-gradient(180deg, #286ee5 0%, #185bc9 100%);
                color: white;
                border: 0;
            }

            .explain-flow {
                display: grid;
                grid-template-columns: repeat(2, minmax(340px, 430px));
                align-items: stretch;
                justify-content: center;
                gap: 2rem;
                margin: 1.25rem 0 1.2rem;
            }

            .model-card {
                border: 1px solid #d5deea;
                border-radius: 8px;
                min-height: 10.8rem;
                width: 100%;
                box-sizing: border-box;
                padding: 1.35rem 1.45rem;
                color: #17223f;
                background: #ffffff;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
            }

            .model-title {
                color: #17223f;
                font-size: 1.16rem;
                font-weight: 780;
                margin-bottom: 0.25rem;
                text-align: center;
            }

            .model-value {
                color: var(--danger);
                font-size: 1.9rem;
                line-height: 1.15;
                font-weight: 820;
                text-align: center;
            }

            .model-meta {
                color: #687386;
                font-size: 0.98rem;
                margin-top: 0.15rem;
                text-align: center;
            }

            .shap-note {
                color: #5b6577;
                font-size: 0.98rem;
                line-height: 1.55;
                margin: 0.3rem 0 0.9rem;
            }

            .note-panel {
                margin-top: 1.1rem;
                color: #687386;
                font-size: 0.98rem;
                box-shadow: 0 8px 20px rgba(17, 39, 76, 0.06);
            }

            [data-testid="stHeader"],
            [data-testid="stToolbar"],
            [data-testid="stDecoration"],
            [data-testid="stStatusWidget"],
            [data-testid="manage-app-button"],
            .stDeployButton,
            #MainMenu,
            footer {
                display: none;
            }

            @media (max-width: 760px) {
                .block-container {
                    padding-left: 1rem;
                    padding-right: 1rem;
                }

                .hero h1 {
                    font-size: 2.1rem;
                }

                .hero p {
                    font-size: 1rem;
                }

                [data-testid="stForm"],
                .note-panel {
                    padding: 1.1rem;
                }

                .explain-flow {
                    grid-template-columns: 1fr;
                    gap: 0.8rem;
                }
            }
        </style>
        """
    )


@st.cache_resource(show_spinner="Loading PCC random forest models...")
def load_models() -> dict[str, dict[str, Any]]:
    loaded: dict[str, dict[str, Any]] = {}
    for key, spec in MODEL_SPECS.items():
        pipeline = joblib.load(spec["model_path"])
        loaded[key] = {
            **spec,
            "pipeline": pipeline,
            "cutoff": float(spec["cutoff"]),
        }
        if key == "plos_conditional":
            loaded[key]["shap_explainer"] = shap.TreeExplainer(pipeline.named_steps["model"])
    return loaded


def model_row(model: Any, values: dict[str, float]) -> pd.DataFrame:
    features = list(model.feature_names_in_)
    return pd.DataFrame([{feature: float(values[feature]) for feature in features}])


def positive_probability(model: Any, values: dict[str, float]) -> float:
    row = model_row(model, values)
    positive_index = list(model.classes_).index(1)
    return float(model.predict_proba(row)[0][positive_index])


def values_with_complication_state(values: dict[str, float], state: int) -> dict[str, float]:
    return {**values, "complication_state": float(state)}


def pcc_predictions(models: dict[str, dict[str, Any]], values: dict[str, float]) -> dict[str, float]:
    complication_probability = positive_probability(models["complication"]["pipeline"], values)
    plos_if_no_complication = positive_probability(
        models["plos_conditional"]["pipeline"],
        values_with_complication_state(values, 0),
    )
    plos_if_complication = positive_probability(
        models["plos_conditional"]["pipeline"],
        values_with_complication_state(values, 1),
    )
    plos_probability = (
        complication_probability * plos_if_complication
        + (1.0 - complication_probability) * plos_if_no_complication
    )
    shap_complication_state = int(complication_probability >= models["complication"]["cutoff"])
    return {
        "complication": complication_probability,
        "plos": float(plos_probability),
        "plos_if_no_complication": plos_if_no_complication,
        "plos_if_complication": plos_if_complication,
        "shap_complication_state": float(shap_complication_state),
    }


def format_percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def risk_level(probability: float, cutoff: float) -> str:
    return "High Risk" if probability >= cutoff else "Low Risk"


def feature_display_name(feature_key: str) -> str:
    if feature_key == "complication_state":
        return "Predicted Complication State"
    for feature in FEATURES:
        if feature["key"] == feature_key:
            return str(feature["label"])
    return feature_key


def feature_unit(feature_key: str) -> str:
    if feature_key == "complication_state":
        return ""
    for feature in FEATURES:
        if feature["key"] == feature_key:
            return str(feature["unit"])
    return ""


def format_feature_value(feature_key: str, value: float) -> str:
    if feature_key == "complication_state":
        return "High Risk" if value >= 0.5 else "Low Risk"
    unit = feature_unit(feature_key)
    formatted = f"{value:.0f}" if feature_key in {"age", "bloodloss", "optime"} else f"{value:.2f}"
    return f"{formatted} {unit}".strip()


def positive_class_shap_values(shap_values: Any, positive_index: int) -> np.ndarray:
    if isinstance(shap_values, list):
        return np.asarray(shap_values[positive_index][0], dtype=float)

    values = np.asarray(shap_values, dtype=float)
    if values.ndim == 3:
        if values.shape[0] == 1:
            return values[0, :, positive_index]
        return values[positive_index, 0, :]
    if values.ndim == 2:
        return values[0, :]
    raise ValueError(f"Unsupported SHAP value shape: {values.shape}")


def positive_expected_value(expected_value: Any, positive_index: int) -> float:
    values = np.asarray(expected_value, dtype=float)
    if values.ndim == 0:
        return float(values)
    return float(values[positive_index])


def los_shap_table(model_info: dict[str, Any], values: dict[str, float]) -> tuple[pd.DataFrame, float]:
    pipeline = model_info["pipeline"]
    row = model_row(pipeline, values)
    transformed = pipeline.named_steps["preprocess"].transform(row)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    positive_index = list(pipeline.classes_).index(1)
    raw_shap_values = model_info["shap_explainer"].shap_values(transformed)
    shap_values = positive_class_shap_values(raw_shap_values, positive_index)
    baseline = positive_expected_value(model_info["shap_explainer"].expected_value, positive_index)

    rows = []
    for feature_key, transformed_value, shap_value in zip(pipeline.feature_names_in_, transformed[0], shap_values):
        rows.append(
            {
                "Feature": feature_display_name(str(feature_key)),
                "Value": format_feature_value(str(feature_key), float(transformed_value)),
                "Contribution": float(shap_value),
                "Contribution (pp)": float(shap_value) * 100.0,
            }
        )

    return pd.DataFrame(rows), baseline


def render_model_cards(
    plos_probability: float,
    plos_cutoff: float,
    complication_probability: float,
    complication_cutoff: float,
) -> None:
    st.html(
        f"""
        <div class="explain-flow">
            <div class="model-card">
                <div class="model-title">PCC PLOS Model</div>
                <div class="model-value">{format_percent(plos_probability)}</div>
                <div class="model-meta">{risk_level(plos_probability, plos_cutoff)}</div>
            </div>
            <div class="model-card">
                <div class="model-title">Complication Model</div>
                <div class="model-value">{format_percent(complication_probability)}</div>
                <div class="model-meta">{risk_level(complication_probability, complication_cutoff)}</div>
            </div>
        </div>
        <div class="shap-note">
            The PCC PLOS probability combines the complication model with the conditional PLOS model. The SHAP contribution plot explains the conditional PLOS model; red features push the prediction toward PLOS, and blue features push it toward non-PLOS.
        </div>
        """
    )


def render_los_shap(model_info: dict[str, Any], values: dict[str, float], shap_complication_state: int) -> None:
    shap_values = values_with_complication_state(values, shap_complication_state)
    table, _baseline = los_shap_table(model_info, shap_values)
    plot_table = table.iloc[::-1]
    colors = ["#f04452" if value > 0 else "#2f6fe4" for value in plot_table["Contribution"]]

    fig, ax = plt.subplots(figsize=(9.8, 4.2), dpi=150)
    labels = [f"{row.Feature} = {row.Value}" for row in plot_table.itertuples(index=False)]
    ax.barh(labels, plot_table["Contribution (pp)"], color=colors, alpha=0.92)
    ax.axvline(0, color="#17223f", linewidth=0.9)
    ax.set_xlabel("SHAP contribution to conditional PLOS probability (percentage points)")
    ax.set_title("PCC PLOS SHAP Contribution Plot", fontsize=12, pad=10)
    ax.grid(axis="x", color="#e2e8f0", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_color("#cfd8e5")
    ax.tick_params(axis="y", labelsize=9)
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True, use_container_width=True)


def default_values() -> dict[str, float]:
    return {str(feature["key"]): float(feature["default"]) for feature in FEATURES}


def render_inputs() -> tuple[dict[str, float], bool]:
    values: dict[str, float] = {}
    st.html('<div class="section-title"><span class="title-icon"></span><span>Input Parameters</span></div>')

    for feature in FEATURES:
        label_col, input_col, unit_col = st.columns([2.9, 1.75, 0.9], vertical_alignment="center")
        with label_col:
            st.html(f'<div class="input-label">{feature["label"]}</div>')
        with input_col:
            values[feature["key"]] = float(
                st.number_input(
                    feature["label"],
                    min_value=float(feature["min"]),
                    max_value=float(feature["max"]),
                    value=float(feature["default"]),
                    step=float(feature["step"]),
                    format=str(feature["format"]),
                    help=feature["help"],
                    label_visibility="collapsed",
                    key=f'input_{feature["key"]}',
                )
            )
        with unit_col:
            st.html(f'<div class="unit-label">{feature["unit"]}</div>')
        st.html('<div class="row-divider"></div>')

    submitted = st.form_submit_button("Predict", use_container_width=True)
    return values, submitted


def main() -> None:
    inject_css()
    models = load_models()

    st.html(
        """
        <div class="hero">
            <h1>PCC Postoperative Risk Prediction Model</h1>
            <p>Predict postoperative LOS &gt;14 days risk and complication risk using a probabilistic classifier chain random forest model</p>
        </div>
        """
    )

    if "prediction_values" not in st.session_state:
        st.session_state["prediction_values"] = default_values()

    with st.form("prediction_form", clear_on_submit=False):
        form_values, submitted = render_inputs()

    if submitted:
        st.session_state["prediction_values"] = form_values

    values = st.session_state["prediction_values"]
    predictions = pcc_predictions(models, values)

    render_model_cards(
        predictions["plos"],
        models["plos_conditional"]["cutoff"],
        predictions["complication"],
        models["complication"]["cutoff"],
    )
    render_los_shap(
        models["plos_conditional"],
        values,
        int(predictions["shap_complication_state"]),
    )

    st.html(
        """
        <div class="note-panel">
            <strong>Note:</strong> These results are for research and decision support only. They do not replace clinical diagnosis or physician judgment and should not be used as the sole basis for diagnosis.
        </div>
        """
    )


if __name__ == "__main__":
    main()
