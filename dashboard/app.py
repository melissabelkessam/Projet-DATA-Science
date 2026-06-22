import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Maintenance Prédictive — EFREI",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

C_BG      = "#F7F9F7"
C_PANEL   = "#FFFFFF"
C_BORDER  = "#E0EBE0"
C_GREEN   = "#1B8A3E"
C_GREEN2  = "#27AE60"
C_GREEN3  = "#A8D5B5"
C_DARK    = "#1A2E1A"
C_GREY    = "#6B7C6B"
C_RED     = "#E74C3C"
C_ORANGE  = "#E67E22"
C_BLUE    = "#2980B9"
C_WHITE   = "#FFFFFF"

st.markdown(f"""
<style>
  .stApp {{ background-color: {C_BG} !important; }}
  .main .block-container {{ background-color: {C_BG} !important; padding-top: 24px; }}

  [data-testid="stSidebar"] {{
    background-color: {C_PANEL} !important;
    border-right: 1px solid {C_BORDER};
    box-shadow: 2px 0 8px rgba(0,0,0,0.04);
  }}
  [data-testid="stSidebar"] * {{ color: {C_DARK} !important; }}

  h1 {{ color: {C_DARK} !important; font-size: 24px !important; font-weight: 800 !important; letter-spacing: -0.5px; }}
  h2, h3 {{ color: {C_DARK} !important; font-weight: 700 !important; }}

  [data-testid="metric-container"] {{
    background: {C_PANEL};
    border: 1px solid {C_BORDER};
    border-top: 3px solid {C_GREEN};
    border-radius: 10px;
    padding: 18px 20px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
  }}
  [data-testid="metric-container"] label {{
    color: {C_GREY} !important;
    font-size: 11px !important;
    text-transform: uppercase;
    letter-spacing: 1px;
    font-weight: 600;
  }}
  [data-testid="metric-container"] [data-testid="stMetricValue"] {{
    color: {C_DARK} !important;
    font-size: 30px !important;
    font-weight: 800 !important;
  }}
  [data-testid="metric-container"] [data-testid="stMetricDelta"] {{
    color: {C_GREEN2} !important;
    font-weight: 600;
  }}

  .stButton > button {{
    background: {C_GREEN} !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 700 !important;
    padding: 12px 32px !important;
    font-size: 15px !important;
    box-shadow: 0 4px 12px rgba(27,138,62,0.25) !important;
    letter-spacing: 0.3px;
  }}
  .stButton > button:hover {{
    background: {C_GREEN2} !important;
    box-shadow: 0 6px 16px rgba(27,138,62,0.35) !important;
    transform: translateY(-1px);
  }}

  hr {{ border: none; border-top: 1px solid {C_BORDER} !important; margin: 24px 0; }}

  .info-card {{
    background: {C_PANEL};
    border: 1px solid {C_BORDER};
    border-left: 4px solid {C_GREEN};
    border-radius: 10px;
    padding: 16px 20px;
    margin: 8px 0;
    font-size: 14px;
    color: {C_DARK};
    box-shadow: 0 2px 6px rgba(0,0,0,0.04);
  }}
  .info-card b {{ color: {C_GREEN}; }}

  .badge {{
    display: inline-block;
    background: {C_GREEN};
    color: white;
    font-size: 10px;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 1.5px;
    text-transform: uppercase;
    margin-left: 10px;
    vertical-align: middle;
  }}

  .page-header {{
    border-bottom: 2px solid {C_BORDER};
    padding-bottom: 14px;
    margin-bottom: 28px;
  }}
  .page-header p {{ color: {C_GREY}; font-size: 14px; margin: 6px 0 0 0; }}

  .nav-label {{
    color: {C_GREY};
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 700;
    margin-bottom: 8px;
    padding-left: 4px;
  }}

  .section-label {{
    color: {C_GREY};
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 1.5px;
    font-weight: 700;
    margin-bottom: 12px;
  }}

  [data-testid="stDataFrame"] {{
    border-radius: 10px;
    overflow: hidden;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
  }}
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

@st.cache_resource
def load_models():
    preprocessor = joblib.load(os.path.join(BASE_DIR, 'models', 'preprocessor.joblib'))
    model        = joblib.load(os.path.join(BASE_DIR, 'models', 'xgboost.joblib'))
    return preprocessor, model

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE_DIR, 'data', 'predictive_maintenance_v3.csv'))

preprocessor, model = load_models()
df = load_data()

FEATURE_NAMES = ['vibration_rms', 'temperature_motor', 'current_phase_avg',
                 'pressure_level', 'rpm', 'hours_since_maintenance',
                 'ambient_temp', 'machine_type_CNC', 'machine_type_Compressor',
                 'machine_type_Pump', 'machine_type_Robotic Arm',
                 'operating_mode_idle', 'operating_mode_normal', 'operating_mode_peak']

NUMERIC_FEATURES = ['vibration_rms', 'temperature_motor', 'current_phase_avg',
                    'pressure_level', 'rpm', 'hours_since_maintenance', 'ambient_temp']

FEATURE_LABELS = {
    'vibration_rms': 'Vibration RMS',
    'temperature_motor': 'Température moteur',
    'current_phase_avg': 'Courant de phase',
    'pressure_level': 'Pression',
    'rpm': 'RPM',
    'hours_since_maintenance': 'Heures depuis maintenance',
    'ambient_temp': 'Température ambiante'
}

def theme(fig, height=400):
    fig.update_layout(
        height=height,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(240,248,240,0.4)',
        font=dict(color=C_DARK, family='Arial'),
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(gridcolor=C_BORDER, linecolor=C_BORDER, tickfont=dict(color=C_GREY)),
        yaxis=dict(gridcolor=C_BORDER, linecolor=C_BORDER, tickfont=dict(color=C_GREY)),
        legend=dict(bgcolor='rgba(255,255,255,0.9)', font=dict(color=C_DARK), bordercolor=C_BORDER, borderwidth=1),
    )
    return fig

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    logo_path = os.path.join(BASE_DIR, 'assets', 'efrei_logo.png')
    if os.path.exists(logo_path):
        st.image(logo_path, width=150)

    st.markdown(f"""
    <div style='padding:14px 0 20px 0; border-bottom:1px solid {C_BORDER}; margin-bottom:24px;'>
      <div style='font-size:17px; font-weight:800; color:{C_DARK}; letter-spacing:-0.3px;'>MaintPredict</div>
      <div style='font-size:11px; color:{C_GREY}; margin-top:3px; text-transform:uppercase; letter-spacing:1px;'>Système IA industriel — M1 DE 2025-26</div>
    </div>
    <div class='nav-label'>Navigation</div>
    """, unsafe_allow_html=True)

    page = st.radio("", [
        "Vue d'ensemble",
        "Analyse des données",
        "Prédiction temps réel",
        "Performance des modèles",
        "Interprétabilité SHAP"
    ], label_visibility="collapsed")

    st.markdown(f"""
    <div style='margin-top:48px; font-size:12px; color:{C_GREY}; border-top:1px solid {C_BORDER}; padding-top:16px; line-height:2;'>
      <div style='color:{C_GREY}; font-size:10px; text-transform:uppercase; letter-spacing:1.5px; font-weight:700; margin-bottom:6px;'>Modèle actif</div>
      Algorithme : <b style='color:{C_GREEN}'>XGBoost</b><br>
      ROC-AUC : <b style='color:{C_GREEN}'>0.9955</b><br>
      Recall : <b style='color:{C_GREEN}'>0.9551</b><br>
      Seuil décision : <b style='color:{C_ORANGE}'>0.70</b>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1
# ═══════════════════════════════════════════════════════════════════════════════
if "Vue d'ensemble" in page:
    st.markdown(f"""
    <div class='page-header'>
      <h1>Tableau de bord — Maintenance Prédictive <span class='badge'>Live</span></h1>
      <p>{len(df):,} enregistrements analysés — Parc de machines industrielles</p>
    </div>""", unsafe_allow_html=True)

    n_pannes = int(df['failure_within_24h'].sum())
    taux = df['failure_within_24h'].mean()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Observations", f"{len(df):,}")
    c2.metric("Pannes détectées", f"{n_pannes:,}", f"{taux:.1%} du parc")
    c3.metric("Types de machines", df['machine_type'].nunique())
    c4.metric("Modes opératoires", df['operating_mode'].nunique())

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        counts = df['failure_within_24h'].value_counts()
        fig = go.Figure(go.Pie(
            values=counts.values, labels=['Machine saine', 'Risque de panne'],
            marker=dict(colors=[C_GREEN2, C_RED]), hole=0.55,
            textfont=dict(color=C_WHITE, size=13)
        ))
        fig.add_annotation(text=f"<b>{taux:.1%}</b><br>pannes", x=0.5, y=0.5,
                           font=dict(size=18, color=C_DARK), showarrow=False)
        fig.update_layout(title=dict(text="Répartition des classes", font=dict(color=C_DARK)),
                          paper_bgcolor='rgba(0,0,0,0)',
                          legend=dict(font=dict(color=C_DARK), bgcolor='rgba(255,255,255,0.9)'),
                          height=350, margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        fm = df.groupby('operating_mode')['failure_within_24h'].mean().reset_index()
        fm.columns = ['Mode', 'Taux']
        fig = px.bar(fm.sort_values('Taux', ascending=True), x='Taux', y='Mode',
                     orientation='h', title="Taux de panne par mode opératoire",
                     color='Taux', color_continuous_scale=[[0,C_GREEN3],[0.5,C_GREEN2],[1,C_RED]],
                     text=fm.sort_values('Taux')['Taux'].map(lambda x: f"{x:.1%}"))
        fig.update_traces(textposition='outside', textfont=dict(color=C_DARK))
        st.plotly_chart(theme(fig, 350), use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        ft = df.groupby('machine_type')['failure_within_24h'].mean().reset_index()
        ft.columns = ['Type', 'Taux']
        fig = px.bar(ft.sort_values('Taux', ascending=False), x='Type', y='Taux',
                     title="Taux de panne par type de machine",
                     color='Taux', color_continuous_scale=[[0,C_GREEN3],[0.5,C_GREEN2],[1,C_RED]],
                     text=ft.sort_values('Taux', ascending=False)['Taux'].map(lambda x: f"{x:.1%}"))
        fig.update_traces(textposition='outside', textfont=dict(color=C_DARK))
        st.plotly_chart(theme(fig, 350), use_container_width=True)

    with col2:
        cross = df.groupby(['machine_type','operating_mode'])['failure_within_24h'].mean().unstack()
        fig = px.imshow(cross, text_auto='.1%',
                        color_continuous_scale=[[0,C_GREEN3],[0.5,C_GREEN2],[1,C_RED]],
                        title="Taux de panne : Machine x Mode opératoire")
        st.plotly_chart(theme(fig, 350), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2
# ═══════════════════════════════════════════════════════════════════════════════
elif "Analyse" in page:
    st.markdown(f"""
    <div class='page-header'>
      <h1>Analyse Exploratoire des Données</h1>
      <p>Distributions, corrélations et comportements des capteurs industriels</p>
    </div>""", unsafe_allow_html=True)

    feature = st.selectbox("Capteur à analyser", NUMERIC_FEATURES,
                           format_func=lambda x: FEATURE_LABELS.get(x, x))
    col1, col2 = st.columns(2)

    with col1:
        fig = px.histogram(df, x=feature, color='failure_within_24h',
                           color_discrete_map={0: C_GREEN2, 1: C_RED},
                           barmode='overlay', opacity=0.75,
                           title=f"Distribution — {FEATURE_LABELS.get(feature, feature)}")
        fig.for_each_trace(lambda t: t.update(name='Sain' if t.name=='0' else 'Panne'))
        st.plotly_chart(theme(fig), use_container_width=True)

    with col2:
        fig = px.box(df, x='failure_within_24h', y=feature,
                     color='failure_within_24h',
                     color_discrete_map={0: C_GREEN2, 1: C_RED},
                     title="Boxplot — OK vs Panne")
        fig.for_each_trace(lambda t: t.update(name='Sain' if t.name=='0' else 'Panne'))
        st.plotly_chart(theme(fig), use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        corr = df[NUMERIC_FEATURES + ['failure_within_24h']].corr()
        fig = px.imshow(corr, text_auto='.2f',
                        color_continuous_scale=[[0,C_BLUE],[0.5,'#FFFFFF'],[1,C_GREEN]],
                        title="Matrice de corrélation des capteurs")
        st.plotly_chart(theme(fig, 450), use_container_width=True)

    with col2:
        st.markdown("<h3>Statistiques descriptives</h3>", unsafe_allow_html=True)
        st.dataframe(df[NUMERIC_FEATURES].describe().round(3), use_container_width=True, height=420)

    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='info-card'>Ratio déséquilibre : <b>5.8:1</b><br>85.2% sain / 14.8% panne</div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='info-card'>Accuracy seule <b>insuffisante</b> — modèle naïf = 85.2% sans détecter une seule panne</div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='info-card'>Métriques retenues : <b>Recall, F1-score, ROC-AUC</b></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3
# ═══════════════════════════════════════════════════════════════════════════════
elif "Prédiction" in page:
    st.markdown(f"""
    <div class='page-header'>
      <h1>Prédiction de Panne en Temps Réel</h1>
      <p>Saisissez les valeurs des capteurs pour évaluer le risque de panne dans les 24h</p>
    </div>""", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown(f"<div class='section-label'>Identification machine</div>", unsafe_allow_html=True)
        machine_type   = st.selectbox("Type de machine", ['CNC', 'Pump', 'Compressor', 'Robotic Arm'])
        operating_mode = st.selectbox("Mode opératoire", ['normal', 'idle', 'peak'])
        vibration      = st.number_input("Vibration RMS (mm/s)", min_value=0.35, max_value=10.0, value=1.5, step=0.05)
    with col2:
        st.markdown(f"<div class='section-label'>Capteurs thermiques et électriques</div>", unsafe_allow_html=True)
        temperature = st.number_input("Température moteur (°C)", min_value=28.0, max_value=95.0, value=50.0, step=0.5)
        current     = st.number_input("Courant phase moyen (A)", min_value=2.2, max_value=35.0, value=8.0, step=0.1)
        pressure    = st.number_input("Pression (bar)", min_value=10.0, max_value=206.0, value=50.0, step=1.0)
    with col3:
        st.markdown(f"<div class='section-label'>Dynamique et historique</div>", unsafe_allow_html=True)
        rpm               = st.number_input("RPM", min_value=124.0, max_value=4098.0, value=1000.0, step=1.0)
        hours_maintenance = st.number_input("Heures depuis maintenance", min_value=0.0, max_value=575.0, value=100.0, step=1.0)
        ambient           = st.number_input("Température ambiante (°C)", min_value=8.0, max_value=18.0, value=13.0, step=0.5)

    st.markdown("---")
    if st.button("Lancer l'analyse de risque", type="primary"):
        input_data = pd.DataFrame({
            'machine_type': [machine_type], 'vibration_rms': [vibration],
            'temperature_motor': [temperature], 'current_phase_avg': [current],
            'pressure_level': [pressure], 'rpm': [rpm],
            'operating_mode': [operating_mode],
            'hours_since_maintenance': [hours_maintenance], 'ambient_temp': [ambient]
        })
        proba      = float(model.predict_proba(preprocessor.transform(input_data))[0][1])
        prediction = 1 if proba >= 0.70 else 0

        col1, col2 = st.columns([1, 1])
        with col1:
            if prediction == 1:
                st.markdown(f"""
                <div style='background:#FEF2F2; border:1px solid #FECACA;
                            border-left:4px solid {C_RED}; border-radius:10px;
                            padding:28px; text-align:center; box-shadow:0 4px 12px rgba(231,76,60,0.08);'>
                  <div style='font-size:13px; font-weight:700; color:{C_RED}; text-transform:uppercase;
                              letter-spacing:2px; margin-bottom:12px;'>Risque de panne détecté</div>
                  <div style='font-size:48px; font-weight:900; color:{C_RED}; line-height:1;'>{proba:.1%}</div>
                  <div style='color:{C_GREY}; font-size:13px; margin-top:10px;'>Probabilité de panne dans les 24h</div>
                  <div style='margin-top:20px; background:#FEF3C7; border-radius:8px;
                              padding:12px; color:{C_ORANGE}; font-size:13px; font-weight:600;'>
                    Intervention de maintenance recommandée dans les 24h
                  </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div style='background:#F0FDF4; border:1px solid #BBF7D0;
                            border-left:4px solid {C_GREEN}; border-radius:10px;
                            padding:28px; text-align:center; box-shadow:0 4px 12px rgba(27,138,62,0.06);'>
                  <div style='font-size:13px; font-weight:700; color:{C_GREEN}; text-transform:uppercase;
                              letter-spacing:2px; margin-bottom:12px;'>Machine en bon état</div>
                  <div style='font-size:48px; font-weight:900; color:{C_GREEN}; line-height:1;'>{proba:.1%}</div>
                  <div style='color:{C_GREY}; font-size:13px; margin-top:10px;'>Probabilité de panne dans les 24h</div>
                  <div style='margin-top:20px; background:#DCFCE7; border-radius:8px;
                              padding:12px; color:{C_GREEN}; font-size:13px; font-weight:600;'>
                    Aucune action immédiate requise
                  </div>
                </div>""", unsafe_allow_html=True)

        with col2:
            bar_color = C_RED if proba >= 0.70 else (C_ORANGE if proba >= 0.30 else C_GREEN2)
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=proba * 100,
                number=dict(suffix="%", font=dict(color=C_DARK, size=36)),
                title=dict(text="Indice de risque", font=dict(color=C_GREY, size=13)),
                gauge=dict(
                    axis=dict(range=[0, 100], tickfont=dict(color=C_GREY)),
                    bar=dict(color=bar_color, thickness=0.3),
                    bgcolor=C_BG, borderwidth=1, bordercolor=C_BORDER,
                    steps=[
                        dict(range=[0,30],   color='rgba(39,174,96,0.10)'),
                        dict(range=[30,70],  color='rgba(230,126,34,0.10)'),
                        dict(range=[70,100], color='rgba(231,76,60,0.10)')
                    ],
                    threshold=dict(line=dict(color=C_ORANGE, width=3), value=70)
                )
            ))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                              font=dict(color=C_DARK),
                              height=280, margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        st.markdown("<h3>Paramètres saisis</h3>", unsafe_allow_html=True)
        st.dataframe(input_data.rename(columns=FEATURE_LABELS), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4
# ═══════════════════════════════════════════════════════════════════════════════
elif "Performance" in page:
    st.markdown(f"""
    <div class='page-header'>
      <h1>Comparaison des Modèles ML et DL</h1>
      <p>Évaluation sur le jeu de test — 4 809 observations (20%)</p>
    </div>""", unsafe_allow_html=True)

    results = {
        'XGBoost':             {'Accuracy':0.9744,'Precision':0.8820,'Recall':0.9551,'F1':0.9171,'ROC-AUC':0.9955},
        'Random Forest':       {'Accuracy':0.9682,'Precision':0.9010,'Recall':0.8820,'F1':0.8914,'ROC-AUC':0.9938},
        'MLP (Deep Learning)': {'Accuracy':0.9310,'Precision':0.7111,'Recall':0.8989,'F1':0.7940,'ROC-AUC':0.9743},
        'Logistic Regression': {'Accuracy':0.9102,'Precision':0.6408,'Recall':0.8947,'F1':0.7468,'ROC-AUC':0.9588},
    }
    cats = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    df_res = pd.DataFrame(results).T.reset_index()
    df_res.columns = ['Modèle'] + cats

    colors_r = [C_GREEN, C_BLUE, C_ORANGE, C_GREY]
    fig = go.Figure()
    for (name, row), color in zip(results.items(), colors_r):
        vals = [row[c] for c in cats]
        fig.add_trace(go.Scatterpolar(
            r=vals + [vals[0]], theta=cats + [cats[0]],
            fill='toself', line=dict(color=color, width=2), name=name,
            fillcolor='rgba(0,0,0,0.05)'
        ))
    fig.update_layout(
        polar=dict(
            bgcolor='rgba(240,248,240,0.4)',
            radialaxis=dict(visible=True, range=[0.6,1], gridcolor=C_BORDER, tickfont=dict(color=C_GREY)),
            angularaxis=dict(gridcolor=C_BORDER, tickfont=dict(color=C_DARK))
        ),
        paper_bgcolor='rgba(0,0,0,0)', font=dict(color=C_DARK),
        title=dict(text="Radar — Comparaison globale des 4 modèles", font=dict(color=C_DARK)),
        legend=dict(bgcolor='rgba(255,255,255,0.9)', font=dict(color=C_DARK),
                    bordercolor=C_BORDER, borderwidth=1), height=420
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        metric = st.selectbox("Métrique à visualiser", cats)
        fig = px.bar(df_res.sort_values(metric, ascending=True),
                     x=metric, y='Modèle', orientation='h',
                     title=f"Comparaison — {metric}",
                     color=metric,
                     color_continuous_scale=[[0,C_GREEN3],[0.7,C_GREEN2],[1,C_GREEN]],
                     text=df_res.sort_values(metric, ascending=True)[metric].map(lambda x: f"{x:.4f}"))
        fig.update_traces(textposition='outside', textfont=dict(color=C_DARK))
        st.plotly_chart(theme(fig, 320), use_container_width=True)
    with col2:
        st.dataframe(df_res.style
                     .highlight_max(axis=0, subset=cats, color='rgba(27,138,62,0.15)')
                     .format({c:'{:.4f}' for c in cats}),
                     use_container_width=True, height=220)
        st.markdown(f"""
        <div class='info-card' style='margin-top:12px;'>
          <b>Modèle retenu : XGBoost</b><br>
          Meilleur ROC-AUC (0.9955), Recall (0.9551) et F1 (0.9171).
          Stable en validation croisée (std Recall = ±0.014).
          Seuil de décision optimisé à <b>0.70</b>.
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<h3>Validation croisée — Stratified K-Fold (5 folds)</h3>", unsafe_allow_html=True)
    cv = pd.DataFrame({
        'Modèle': ['XGBoost', 'Random Forest', 'Logistic Regression'],
        'Recall moyen': [0.9466, 0.8683, 0.8838],
        'Recall std': ['±0.0135', '±0.0113', '±0.0128'],
        'F1 moyen': [0.9127, 0.8932, 0.7454],
        'ROC-AUC moyen': [0.9949, 0.9930, 0.9562],
    })
    st.dataframe(cv, use_container_width=True)

    st.markdown("---")
    st.markdown("<h3>Techniques de gestion du déséquilibre</h3>", unsafe_allow_html=True)
    imb = pd.DataFrame({
        'Technique': ['class_weight (retenu)', 'Random Over-Sampling', 'SMOTE', 'Random Under-Sampling'],
        'Recall': [0.9551, 0.9494, 0.9438, 0.9719],
        'F1': [0.9171, 0.9111, 0.8924, 0.8491],
        'ROC-AUC': [0.9955, 0.9960, 0.9949, 0.9910],
    })
    fig = px.bar(imb, x='Technique', y=['Recall', 'F1', 'ROC-AUC'], barmode='group',
                 title="Comparaison des techniques de rééquilibrage",
                 color_discrete_map={'Recall': C_RED, 'F1': C_GREEN, 'ROC-AUC': C_BLUE})
    st.plotly_chart(theme(fig, 360), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5
# ═══════════════════════════════════════════════════════════════════════════════
elif "SHAP" in page:
    st.markdown(f"""
    <div class='page-header'>
      <h1>Interprétabilité — Modèle XGBoost</h1>
      <p>Comprendre pourquoi le modèle déclenche une alerte de panne</p>
    </div>""", unsafe_allow_html=True)

    importances = model.feature_importances_
    df_imp = pd.DataFrame({'Variable': FEATURE_NAMES, 'Importance': importances}).sort_values('Importance', ascending=True)
    fig = px.bar(df_imp, x='Importance', y='Variable', orientation='h',
                 title="Feature Importance — XGBoost (réduction d'impureté Gini)",
                 color='Importance',
                 color_continuous_scale=[[0,C_GREEN3],[0.5,C_GREEN2],[1,C_GREEN]])
    fig.update_traces(texttemplate='%{x:.3f}', textposition='outside', textfont=dict(color=C_DARK, size=11))
    st.plotly_chart(theme(fig, 500), use_container_width=True)

    st.markdown("---")
    shap_s = os.path.join(BASE_DIR, 'reports', 'shap_summary.png')
    shap_a = os.path.join(BASE_DIR, 'reports', 'shap_analysis.png')
    if os.path.exists(shap_s) or os.path.exists(shap_a):
        col1, col2 = st.columns(2)
        if os.path.exists(shap_s):
            with col1:
                st.markdown("<h3>SHAP Summary Plot</h3>", unsafe_allow_html=True)
                st.image(shap_s, use_container_width=True)
        if os.path.exists(shap_a):
            with col2:
                st.markdown("<h3>SHAP — Analyse détaillée</h3>", unsafe_allow_html=True)
                st.image(shap_a, use_container_width=True)

    st.markdown("---")
    st.markdown("<h3>Interprétation métier des variables clés</h3>", unsafe_allow_html=True)
    variables = [
        ("Température moteur", "Signal thermique principal. Une température supérieure à 70°C est fortement associée à un risque de panne — reflète la charge et l'usure du moteur."),
        ("Vibration RMS", "Des vibrations supérieures à 4 mm/s révèlent une dégradation mécanique : déséquilibre, usure des roulements ou défaut d'alignement."),
        ("RPM", "Une vitesse anormalement élevée ou instable indique une surcharge mécanique ou un défaut de régulation."),
        ("Courant de phase", "Un courant excessif signale une surcharge électrique ou un court-circuit partiel en développement."),
        ("Heures depuis maintenance", "Plus la durée depuis la dernière intervention est longue, plus le risque s'accumule. Variable temporelle clé dans la prédiction."),
    ]
    for i in range(0, len(variables), 2):
        c1, c2 = st.columns(2)
        with c1:
            t, d = variables[i]
            st.markdown(f"<div class='info-card'><b>{t}</b><br><span style='color:{C_GREY};font-size:13px;line-height:1.6;'>{d}</span></div>", unsafe_allow_html=True)
        if i+1 < len(variables):
            with c2:
                t, d = variables[i+1]
                st.markdown(f"<div class='info-card'><b>{t}</b><br><span style='color:{C_GREY};font-size:13px;line-height:1.6;'>{d}</span></div>", unsafe_allow_html=True)
