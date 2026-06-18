import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBClassifier

st.set_page_config(
    page_title="Maintenance Predictive Industrielle",
    page_icon="",
    layout="wide"
)

COLOR_BLUE  = "#163767"
COLOR_RED   = "#E53935"
COLOR_GREEN = "#43A047"

@st.cache_resource
def load_models():
    preprocessor = joblib.load('../models/preprocessor.joblib')
    model = joblib.load('../models/xgboost.joblib')
    return preprocessor, model

@st.cache_data
def load_data():
    df = pd.read_csv('../data/predictive_maintenance_v3.csv')
    return df

preprocessor, model = load_models()
df = load_data()

FEATURE_NAMES = ['vibration_rms', 'temperature_motor', 'current_phase_avg',
                 'pressure_level', 'rpm', 'hours_since_maintenance',
                 'ambient_temp', 'machine_type_CNC', 'machine_type_Compressor',
                 'machine_type_Pump', 'machine_type_Robotic Arm',
                 'operating_mode_idle', 'operating_mode_normal', 'operating_mode_peak']

NUMERIC_FEATURES = ['vibration_rms', 'temperature_motor', 'current_phase_avg',
                    'pressure_level', 'rpm', 'hours_since_maintenance', 'ambient_temp']

st.sidebar.title("Navigation")
page = st.sidebar.radio("", [
    "Vue d'ensemble",
    "Analyse des donnees",
    "Prediction en temps reel",
    "Performance des modeles",
    "Interpretabilite SHAP"
])

# ============================================================
# PAGE 1 - Vue d'ensemble
# ============================================================
if page == "Vue d'ensemble":
    st.title("Systeme de Maintenance Predictive Industrielle")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total observations", f"{len(df):,}")
    col2.metric("Taux de panne", f"{df['failure_within_24h'].mean():.1%}")
    col3.metric("Types de machines", df['machine_type'].nunique())
    col4.metric("Modes operatoires", df['operating_mode'].nunique())

    st.markdown("---")
    st.subheader("Distribution des pannes")

    col1, col2 = st.columns(2)
    with col1:
        counts = df['failure_within_24h'].value_counts()
        fig = px.pie(values=counts.values,
                     names=['Pas de panne', 'Panne'],
                     color_discrete_sequence=[COLOR_GREEN, COLOR_RED],
                     title="Repartition des classes")
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        failure_by_mode = df.groupby('operating_mode')['failure_within_24h'].mean().reset_index()
        fig = px.bar(failure_by_mode, x='operating_mode', y='failure_within_24h',
                     title="Taux de panne par mode operatoire",
                     color_discrete_sequence=[COLOR_BLUE])
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Taux de panne par type de machine")
    failure_by_type = df.groupby('machine_type')['failure_within_24h'].mean().reset_index()
    fig = px.bar(failure_by_type, x='machine_type', y='failure_within_24h',
                 color='machine_type', title="Taux de panne par type de machine")
    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# PAGE 2 - Analyse des donnees
# ============================================================
elif page == "Analyse des donnees":
    st.title("Analyse Exploratoire des Donnees")
    st.markdown("---")

    feature = st.selectbox("Selectionnez un capteur", NUMERIC_FEATURES)

    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x=feature, color='failure_within_24h',
                           color_discrete_map={0: COLOR_GREEN, 1: COLOR_RED},
                           title=f"Distribution de {feature} par classe",
                           barmode='overlay', opacity=0.7)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fig = px.box(df, x='failure_within_24h', y=feature,
                     color='failure_within_24h',
                     color_discrete_map={0: COLOR_GREEN, 1: COLOR_RED},
                     title=f"Boxplot {feature} — OK vs Panne")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Matrice de correlation")
    corr = df[NUMERIC_FEATURES + ['failure_within_24h']].corr()
    fig = px.imshow(corr, text_auto='.2f', color_continuous_scale='RdBu_r',
                    title="Heatmap de correlation")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Statistiques descriptives")
    st.dataframe(df[NUMERIC_FEATURES].describe().round(2), use_container_width=True)

# ============================================================
# PAGE 3 - Prediction en temps reel
# ============================================================
elif page == "Prediction en temps reel":
    st.title("Prediction de Panne en Temps Reel")
    st.markdown("---")
    st.subheader("Saisissez les valeurs des capteurs")

    col1, col2, col3 = st.columns(3)

    with col1:
        machine_type = st.selectbox("Type de machine", ['CNC', 'Pump', 'Compressor', 'Robotic Arm'])
        operating_mode = st.selectbox("Mode operatoire", ['normal', 'idle', 'peak'])
        vibration = st.slider("Vibration RMS", 0.35, 10.0, 1.5)

    with col2:
        temperature = st.slider("Temperature moteur (C)", 28.0, 95.0, 50.0)
        current = st.slider("Courant phase moyen (A)", 2.2, 35.0, 8.0)
        pressure = st.slider("Pression (bar)", 10.0, 206.0, 50.0)

    with col3:
        rpm = st.slider("RPM", 124.0, 4098.0, 1000.0)
        hours_maintenance = st.slider("Heures depuis maintenance", 0.0, 575.0, 100.0)
        ambient = st.slider("Temperature ambiante (C)", 8.0, 18.0, 13.0)

    if st.button("Lancer la prediction", type="primary"):
        input_data = pd.DataFrame({
            'machine_type': [machine_type],
            'vibration_rms': [vibration],
            'temperature_motor': [temperature],
            'current_phase_avg': [current],
            'pressure_level': [pressure],
            'rpm': [rpm],
            'operating_mode': [operating_mode],
            'hours_since_maintenance': [hours_maintenance],
            'ambient_temp': [ambient]
        })

        input_processed = preprocessor.transform(input_data)
        proba = float(model.predict_proba(input_processed)[0][1])
        prediction = 1 if proba >= 0.70 else 0

        st.markdown("---")
        if prediction == 1:
            st.error(f"RISQUE DE PANNE DETECTE — Probabilite : {proba:.1%}")
            st.warning("Action recommandee : Planifier une intervention de maintenance dans les 24h")
        else:
            st.success(f"Machine en bon etat — Probabilite de panne : {proba:.1%}")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=proba * 100,
            title={'text': "Risque de panne (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': COLOR_RED if proba >= 0.70 else COLOR_GREEN},
                'steps': [
                    {'range': [0, 30], 'color': '#e8f5e9'},
                    {'range': [30, 70], 'color': '#fff3e0'},
                    {'range': [70, 100], 'color': '#ffebee'}
                ],
                'threshold': {'line': {'color': 'black', 'width': 4}, 'value': 70}
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Valeurs saisies")
        st.dataframe(input_data, use_container_width=True)

# ============================================================
# PAGE 4 - Performance des modeles
# ============================================================
elif page == "Performance des modeles":
    st.title("Comparaison des Modeles")
    st.markdown("---")

    results = {
        'XGBoost':             {'Accuracy': 0.9744, 'Precision': 0.8820, 'Recall': 0.9551, 'F1': 0.9171, 'ROC-AUC': 0.9955},
        'Random Forest':       {'Accuracy': 0.9682, 'Precision': 0.9010, 'Recall': 0.8820, 'F1': 0.8914, 'ROC-AUC': 0.9938},
        'MLP':                 {'Accuracy': 0.9310, 'Precision': 0.7111, 'Recall': 0.8989, 'F1': 0.7940, 'ROC-AUC': 0.9743},
        'Logistic Regression': {'Accuracy': 0.9102, 'Precision': 0.6408, 'Recall': 0.8947, 'F1': 0.7468, 'ROC-AUC': 0.9588},
    }

    df_results = pd.DataFrame(results).T.reset_index()
    df_results.columns = ['Modele'] + list(df_results.columns[1:])

    st.dataframe(
        df_results.style.highlight_max(
            axis=0, subset=['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
        ),
        use_container_width=True
    )

    metric = st.selectbox("Metrique a visualiser", ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC'])
    fig = px.bar(df_results, x='Modele', y=metric,
                 color='Modele', title=f"Comparaison — {metric}",
                 color_discrete_sequence=[COLOR_BLUE, COLOR_RED, COLOR_GREEN, '#FF8C00'])
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Validation croisee Stratified K-Fold (5 folds)")
    cv_data = {
        'Modele': ['XGBoost', 'Random Forest', 'Logistic Regression'],
        'Recall moyen': [0.9466, 0.8683, 0.8838],
        'Recall std': [0.0135, 0.0113, 0.0128],
        'F1 moyen': [0.9127, 0.8932, 0.7454],
        'ROC-AUC moyen': [0.9949, 0.9930, 0.9562],
    }
    st.dataframe(pd.DataFrame(cv_data), use_container_width=True)

# ============================================================
# PAGE 5 - Interpretabilite SHAP
# ============================================================
elif page == "Interpretabilite SHAP":
    st.title("Interpretabilite du Modele XGBoost")
    st.markdown("---")

    importances = model.feature_importances_
    df_imp = pd.DataFrame({
        'Feature': FEATURE_NAMES,
        'Importance': importances
    }).sort_values('Importance', ascending=True)

    fig = px.bar(df_imp, x='Importance', y='Feature',
                 orientation='h',
                 title='Feature Importance — XGBoost (Gini)',
                 color='Importance',
                 color_continuous_scale='Blues')
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    st.subheader("Top 5 variables les plus importantes")

    col1, col2 = st.columns(2)
    with col1:
        st.info("**operating_mode_normal** : Le mode operatoire est le signal le plus discriminant du modele.")
        st.info("**temperature_motor** : Une temperature elevee est fortement associee a un risque de panne.")
        st.info("**rpm** : Une vitesse de rotation anormale indique une surcharge mecanique.")
    with col2:
        st.info("**vibration_rms** : Des vibrations elevees revelent une degradation des composants.")
        st.info("**operating_mode_peak** : Le mode peak concentre davantage de pannes que les autres modes.")

    st.markdown("---")
    st.subheader("Interpretation metier")
    st.markdown("""
    Le modele XGBoost identifie les pannes principalement a travers :
    - Les **conditions operatoires** (mode normal/peak) qui definissent le regime de travail de la machine
    - La **temperature moteur** qui reflète la charge thermique et l'usure
    - Le **rpm** et la **vibration** qui capturent la dynamique mecanique
    
    Un responsable maintenance peut ainsi prioriser la surveillance des machines en mode peak 
    presentant des temperatures et vibrations elevees.
    """)