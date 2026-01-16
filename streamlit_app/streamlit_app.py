import os
import sys
import json
import base64
import requests
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from typing import Any
from pathlib import Path

# =========================================================
# 0. PATH SETUP
# =========================================================

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# =========================================================
# 1. INIT CONFIG
# =========================================================

from config.api_config import (
    API_BASE_URL,
    PREDICT_ENDPOINT,
    SHAP_LOCAL_ENDPOINT,
    SHAP_GLOBAL_ENDPOINT,
    HEALTH_ENDPOINT
)

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(APP_DIR, "data", "df_test_sample.csv")

# =========================================================
# 2. LOAD FEATURES
# =========================================================

APP_DIR = Path(__file__).resolve().parent

FEATURES_PATH = APP_DIR / "models" / "feature_names.json"

with open(FEATURES_PATH, "r") as f:
    FEATURE_NAMES = json.load(f)

# =========================================================
# 3. UTILS
# =========================================================

def clean_for_api(d: dict) -> dict:
    clean = {}
    for k, v in d.items():
        try:
            val = float(v)
            if not np.isfinite(val):
                val = 0.0
        except Exception:
            val = 0.0
        clean[k] = val
    return clean


@st.cache_data
def load_test_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    return df.reindex(columns=FEATURE_NAMES, fill_value=0)


def load_image_base64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode()


def safe_get(sample: pd.Series, col: str, default: Any = None) -> Any:
    return sample[col] if col in sample.index else default

def interpret_risk(proba, thr):
    delta = proba - thr

    if delta > 0.10:
        return (
            "Risque nettement supérieur au seuil. "
            "La décision de refus est clairement justifiée."
        )
    elif delta > 0:
        return (
            "Risque légèrement supérieur au seuil. "
            "La situation est borderline et pourrait être réévaluée."
        )
    elif delta > -0.05:
        return (
            "Risque proche du seuil mais inférieur. "
            "La décision reste favorable avec prudence."
        )
    else:
        return (
            "Risque nettement inférieur au seuil. "
            "La décision d’acceptation est confortable."
        )



# =========================================================
# 4. PAGE SETUP
# =========================================================

st.set_page_config(
    page_title="Décision de crédit – Fiche client",
    layout="wide"
)

st.markdown("# Décision de crédit – Fiche client")
st.markdown(
    "Outil d’aide à la décision à destination des **chargés de relation client**."
)

# =========================================================
# 5. LOAD DATA
# =========================================================

df_test = load_test_data(DATA_PATH)
n_rows = len(df_test)
max_idx = n_rows - 1

# =========================================================
# 6. SESSION STATE
# =========================================================

st.session_state.setdefault("idx", 0)
st.session_state.setdefault("prediction_result", None)
st.session_state.setdefault("shap_result", None)

# =========================================================
# 7. SIDEBAR – BANNIÈRE + PROFIL CLIENT
# =========================================================

BANNER_PATH = os.path.join(APP_DIR, "banner.png")
if os.path.exists(BANNER_PATH):
    img64 = load_image_base64(BANNER_PATH)
    st.sidebar.markdown(
        f"""
        <div style="text-align:center; margin-bottom:15px;">
            <img src="data:image/png;base64,{img64}" style="width:100%;">
        </div>
        """,
        unsafe_allow_html=True
    )

st.sidebar.title("Sélection du client")

idx_text = st.sidebar.text_input(
    "Index client",
    value=str(st.session_state.idx),
    help=f"Entier entre 0 et {max_idx}"
)

if st.sidebar.button("Charger le client"):
    try:
        st.session_state.idx = max(0, min(int(idx_text), max_idx))
        st.session_state.prediction_result = None
        st.session_state.shap_result = None
    except ValueError:
        st.sidebar.error("Index invalide")

sample = df_test.iloc[st.session_state.idx]

# =========================================================
# PROFIL CLIENT (BASÉ SUR VARIABLES RÉELLES)
# =========================================================

st.sidebar.markdown("### 👤 Profil client")

if "CODE_GENDER_F" in sample.index:
    sexe = "Femme" if sample["CODE_GENDER_F"] == 1 else "Homme"
    st.sidebar.markdown(f"**Sexe :** {sexe}")

if "NAME_FAMILY_STATUS_Married" in sample.index:
    statut = "Marié(e)" if sample["NAME_FAMILY_STATUS_Married"] == 1 else "Autre"
    st.sidebar.markdown(f"**Situation familiale :** {statut}")

if "NAME_INCOME_TYPE_Working" in sample.index:
    pro = "En activité" if sample["NAME_INCOME_TYPE_Working"] == 1 else "Autre"
    st.sidebar.markdown(f"**Situation professionnelle :** {pro}")

if "NAME_EDUCATION_TYPE_Higher_education" in sample.index:
    edu = "Études supérieures" if sample["NAME_EDUCATION_TYPE_Higher_education"] == 1 else "Autre"
    st.sidebar.markdown(f"**Niveau d’études :** {edu}")

st.sidebar.markdown("---")
st.sidebar.markdown("### 💼 Stabilité & historique")

if "DAYS_EMPLOYED" in sample.index:
    st.sidebar.markdown(
        f"**Ancienneté dans l’emploi :** {int(abs(sample['DAYS_EMPLOYED']) // 365)} ans"
    )

st.sidebar.markdown("---")
st.sidebar.markdown("### 💰 Situation financière")

if "ANNUITY_INCOME_PERC" in sample.index:
    st.sidebar.markdown(
        f"**Part du revenu consacrée au crédit :** {sample['ANNUITY_INCOME_PERC']:.2%}"
    )

# =========================================================
# 8. API STATUS
# =========================================================

st.markdown("## 🔌 Statut du service")

with st.spinner("Réveil de l’API en cours…"):
    r = requests.get(HEALTH_ENDPOINT, timeout=60)

if r.status_code == 200:
    st.success("API opérationnelle")
else:
    st.error("API non joignable")
    st.stop()

# =========================================================
# 9. PREDICTION
# =========================================================

col_title, col_button = st.columns([4, 1])

with col_title:
    st.markdown("## Décision de crédit")

with col_button:
    st.markdown(
        """
        <div style="margin-top: 20px;">
        """,
        unsafe_allow_html=True
    )
    launch_analysis = st.button("Lancer l’analyse", use_container_width=True)

if launch_analysis:
    with st.spinner("Calcul en cours…"):
        try:
            payload = {"data": clean_for_api(sample.to_dict())}

            pred_resp = requests.post(
                PREDICT_ENDPOINT,
                json=payload,
                timeout=60
            )

            shap_resp = requests.post(
                SHAP_LOCAL_ENDPOINT,
                json=payload,
                timeout=60
            )

            st.session_state.prediction_result = pred_resp.json()
            st.session_state.shap_result = shap_resp.json()

        except Exception as e:
            st.error(f"Erreur API : {e}")
# =========================================================
# 10. AFFICHAGE DES RÉSULTATS
# =========================================================

if st.session_state.prediction_result:

    pred = st.session_state.prediction_result
    shap_api = st.session_state.shap_result

    if shap_api is None:
        st.info("Veuillez lancer l’analyse pour afficher l’explicabilité.")
        st.stop()

    proba = pred["probability_default"]
    decision = pred["decision"]
    thr = pred["threshold_used"]

    COLOR_RISK_HIGH = "#B00020"   # rouge sombre (contrasté)
    COLOR_RISK_LOW  = "#0B6E4F"   # vert sombre

    color = COLOR_RISK_HIGH if decision == 1 else COLOR_RISK_LOW
    label = "RISQUE ÉLEVÉ" if decision == 1 else "RISQUE FAIBLE"

    delta = proba - thr
    interpretation = interpret_risk(proba, thr)

    st.markdown(
    f"""
    <div style="
    padding:20px;
    border-radius:12px;
    background:#f9f9f9;
    border-left:8px solid {color};
    margin-bottom:20px;
    ">
    <h3 style="margin:0;color:{color};">{label}</h3>

    <p style="margin-top:10px;">
    <strong>Probabilité de défaut :</strong> {proba:.1%}<br>
    <strong>Seuil d’acceptation :</strong> {thr:.1%}<br>
    <strong>Écart au seuil :</strong> {delta:+.1%}
    </p>

    <div style="
    margin-top:10px;
    padding:10px;
    background:#ffffff;
    border-radius:8px;
    border:1px solid #ddd;
    ">
    <strong>Interprétation :</strong><br>
    {interpretation}
    </div>
    </div>
    """,
    unsafe_allow_html=True
)


    shap_df = pd.DataFrame({
        "feature": shap_api["feature_names"],
        "contribution": shap_api["shap_values"],
        "value": shap_api["features"]
    }).assign(impact=lambda x: x["contribution"].abs()).sort_values("impact", ascending=False)

    tab_profile, tab_bivariate, tab_local, tab_global = st.tabs([
        "Comparaison client / population",
        "Analyse bi-variée",
        "Explication de la décision",
        "Importance globale"
    ])


    # =========================================================
    # TAB 1 – COMPARAISON CLIENT / POPULATION
    # =========================================================
    with tab_profile:
        st.markdown("### Positionnement du client par rapport aux autres clients")

        numeric_cols = df_test.select_dtypes(include=["float", "int"]).columns.tolist()

        selected_var = st.selectbox(
            "Choisir une variable à comparer",
            numeric_cols,
            help="Comparer la valeur du client à l’ensemble des clients"
        )

        client_value = sample[selected_var]

        col_left, col_center, col_right = st.columns([1, 3, 1])

        with col_center:

            fig, ax = plt.subplots(figsize=(5, 3))

            ax.hist(
                df_test[selected_var].dropna(),
                bins=30,
                alpha=0.7,
                label="Population"
            )

            if pd.notna(client_value):
                ax.axvline(
                    client_value,
                    color="red",
                    linewidth=2,
                    label="Client"
                )
            else:
                st.warning(
                    "⚠️ La valeur de cette variable n’est pas renseignée pour ce client."
                )

            ax.set_title(f"Distribution de {selected_var}")
            ax.set_xlabel(selected_var)
            ax.set_ylabel("Nombre de clients")
            ax.legend()

            st.pyplot(fig)

            st.caption(
                "Histogramme représentant la distribution de la variable "
                "dans la population. La position du client est indiquée par "
                "une ligne verticale (rouge) lorsque la donnée est disponible."
            )

            plt.close(fig)

        with st.expander("Voir les valeurs sous forme de tableau"):
            st.dataframe(
                df_test[[selected_var]].describe().T
                .rename(columns={
                    "count": "Nombre de valeurs",
                    "std": "Ecart-type",
                    "mean": "Moyenne",
                    "min": "Minimum",
                    "max": "Maximum"
                }),
                use_container_width=True
            )

    # =========================================================
    # TAB 2 – ANALYSE BI-VARIÉE
    # =========================================================
    with tab_bivariate:
        st.markdown("### Analyse bi-variée entre deux variables")

        numeric_cols = df_test.select_dtypes(include=["float", "int"]).columns.tolist()

        col_x, col_y = st.columns(2)

        with col_x:
            var_x = st.selectbox(
                "Variable X",
                numeric_cols,
                key="bivariate_x"
            )

        with col_y:
            var_y = st.selectbox(
                "Variable Y",
                numeric_cols,
                index=1 if len(numeric_cols) > 1 else 0,
                key="bivariate_y"
            )

        client_x = sample[var_x]
        client_y = sample[var_y]

        col_left, col_center, col_right = st.columns([1, 3, 1])

        with col_center:

            fig, ax = plt.subplots(figsize=(5, 3))

            # Population
            ax.scatter(
                df_test[var_x],
                df_test[var_y],
                alpha=0.3,
                label="Population",
                color="#7f8c8d"
            )

            # Client
            if pd.notna(client_x) and pd.notna(client_y):
                ax.scatter(
                    client_x,
                    client_y,
                    color="#B00020",
                    s=120,
                    edgecolor="black",
                    label="Client"
                )
            else:
                st.warning(
                    "⚠️ Une des deux variables n’est pas renseignée pour ce client. "
                    "Le point client ne peut pas être affiché."
                )

            ax.set_xlabel(var_x)
            ax.set_ylabel(var_y)
            ax.set_title(f"Relation entre {var_x} et {var_y}")
            ax.legend()

            st.pyplot(fig)

            st.caption(
                "Ce graphique permet d’analyser la relation entre deux variables "
                "dans la population. Le client est positionné lorsqu’il dispose des deux valeurs."
            )

            plt.close(fig)


    # =========================================================
    # TAB 3 – EXPLICATION LOCALE (SHAP)
    # =========================================================
    with tab_local:
        st.markdown("### Principales raisons de la décision")

        top = shap_df.head(8)

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(
            top["feature"][::-1],
            top["contribution"][::-1],
            color=["#c0392b" if v > 0 else "#27ae60" for v in top["contribution"][::-1]]
        )

        ax.set_xlabel("Impact sur le risque")
        ax.set_title("Variables les plus influentes")

        st.pyplot(fig)

        st.caption(
            "Barres rouges : variables qui augmentent le risque de défaut. "
            "Barres vertes : variables qui réduisent le risque."
        )

        with st.expander("Voir l’explication sous forme de tableau"):
            st.dataframe(
                top
                .drop(columns=["impact"])
                .rename(columns={
                    "feature": "Variable",
                    "contribution": "Impact sur le risque",
                    "value": "Valeur client"
                }),
                use_container_width=True
            )

        plt.close(fig)

        positive = top[top["contribution"] > 0]["feature"].tolist()
        negative = top[top["contribution"] < 0]["feature"].tolist()

        st.markdown("#### Interprétation métier")

        if decision == 1:
            st.write(
                f"La décision de refus est principalement liée à : "
                f"{', '.join(positive[:3])}."
            )
        else:
            st.write(
                f"La décision d’acceptation est notamment expliquée par : "
                f"{', '.join(negative[:3])}."
            )

    # =========================================================
    # TAB 4 – IMPORTANCE GLOBALE
    # =========================================================
    with tab_global:
        st.markdown("### Variables influentes sur l’ensemble des clients")

        shap_global_api = requests.get(SHAP_GLOBAL_ENDPOINT, timeout=60).json()

        df_global = (
            pd.DataFrame({
                "Variable": shap_global_api["feature_names"],
                "Importance": np.abs(shap_global_api["shap_values"]).mean(axis=0)
            })
            .sort_values("Importance", ascending=False)
            .head(15)
        )

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.barh(df_global["Variable"][::-1], df_global["Importance"][::-1])
        ax.set_xlabel("Importance moyenne")

        st.pyplot(fig)

        st.caption(
            "Classement des variables selon leur importance moyenne "
            "dans les décisions du modèle."
        )

        with st.expander("Voir les importances sous forme de tableau"):
            st.dataframe(df_global, use_container_width=True)

        plt.close(fig)

# =========================================================
# FOOTER
# =========================================================

st.markdown("<hr><small>Dashboard connecté à l’API de scoring – usage interne</small>", unsafe_allow_html=True)
