import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
import requests

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
COUT_INTERVENTION_IA = 800

st.markdown(f"""
<style>
  .stApp {{ background-color: {C_BG} !important; }}
  .main .block-container {{ background-color: {C_BG} !important; padding-top: 24px; }}
  [data-testid="stSidebar"] {{ background-color: {C_PANEL} !important; border-right: 1px solid {C_BORDER}; box-shadow: 2px 0 8px rgba(0,0,0,0.04); }}
  [data-testid="stSidebar"] * {{ color: {C_DARK} !important; }}
  h1 {{ color: {C_DARK} !important; font-size: 24px !important; font-weight: 800 !important; letter-spacing: -0.5px; }}
  h2, h3 {{ color: {C_DARK} !important; font-weight: 700 !important; }}
  [data-testid="metric-container"] {{ background: {C_PANEL}; border: 1px solid {C_BORDER}; border-top: 3px solid {C_GREEN}; border-radius: 10px; padding: 18px 20px; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }}
  [data-testid="metric-container"] label {{ color: {C_GREY} !important; font-size: 11px !important; text-transform: uppercase; letter-spacing: 1px; font-weight: 600; }}
  [data-testid="metric-container"] [data-testid="stMetricValue"] {{ color: {C_DARK} !important; font-size: 30px !important; font-weight: 800 !important; }}
  [data-testid="metric-container"] [data-testid="stMetricDelta"] {{ color: {C_GREEN2} !important; font-weight: 600; }}
  .stButton > button {{ background: {C_GREEN} !important; color: white !important; border: none !important; border-radius: 8px !important; font-weight: 700 !important; padding: 12px 32px !important; font-size: 15px !important; box-shadow: 0 4px 12px rgba(27,138,62,0.25) !important; }}
  .stButton > button:hover {{ background: {C_GREEN2} !important; }}
  hr {{ border: none; border-top: 1px solid {C_BORDER} !important; margin: 24px 0; }}
  .info-card {{ background: {C_PANEL}; border: 1px solid {C_BORDER}; border-left: 4px solid {C_GREEN}; border-radius: 10px; padding: 16px 20px; margin: 8px 0; font-size: 14px; color: {C_DARK}; box-shadow: 0 2px 6px rgba(0,0,0,0.04); }}
  .info-card b {{ color: {C_GREEN}; }}
  .badge {{ display: inline-block; background: {C_GREEN}; color: white; font-size: 10px; font-weight: 700; padding: 3px 10px; border-radius: 20px; letter-spacing: 1.5px; text-transform: uppercase; margin-left: 10px; vertical-align: middle; }}
  .api-badge {{ display: inline-block; background: {C_BLUE}; color: white; font-size: 10px; font-weight: 700; padding: 3px 10px; border-radius: 20px; letter-spacing: 1.5px; text-transform: uppercase; margin-left: 10px; vertical-align: middle; }}
  .page-header {{ border-bottom: 2px solid {C_BORDER}; padding-bottom: 14px; margin-bottom: 28px; }}
  .page-header p {{ color: {C_GREY}; font-size: 14px; margin: 6px 0 0 0; }}
  .nav-label {{ color: {C_GREY}; font-size: 10px; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700; margin-bottom: 8px; padding-left: 4px; }}
  .section-label {{ color: {C_GREY}; font-size: 10px; text-transform: uppercase; letter-spacing: 1.5px; font-weight: 700; margin-bottom: 12px; }}
  [data-testid="stDataFrame"] {{ border-radius: 10px; overflow: hidden; box-shadow: 0 2px 8px rgba(0,0,0,0.05); }}
  .kpi-alert {{ background: #FEF2F2; border: 1px solid #FECACA; border-top: 4px solid {C_RED}; border-radius: 10px; padding: 20px; text-align: center; }}
  .kpi-ok {{ background: #F0FDF4; border: 1px solid #BBF7D0; border-top: 4px solid {C_GREEN}; border-radius: 10px; padding: 20px; text-align: center; }}
  .kpi-warn {{ background: #FFFBEB; border: 1px solid #FDE68A; border-top: 4px solid {C_ORANGE}; border-radius: 10px; padding: 20px; text-align: center; }}
</style>
""", unsafe_allow_html=True)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_API_URL = "http://localhost:8000"

@st.cache_resource
def load_models():
    preprocessor = joblib.load(os.path.join(BASE_DIR, 'models', 'preprocessor.joblib'))
    model        = joblib.load(os.path.join(BASE_DIR, 'models', 'xgboost.joblib'))
    return preprocessor, model

@st.cache_resource
def load_rul_model():
    rul_path      = os.path.join(BASE_DIR, 'models', 'rul_model.joblib')
    rul_prep_path = os.path.join(BASE_DIR, 'models', 'rul_preprocessor.joblib')
    if os.path.exists(rul_path) and os.path.exists(rul_prep_path):
        return joblib.load(rul_path), joblib.load(rul_prep_path)
    elif os.path.exists(rul_path):
        # Fallback : si rul_preprocessor absent, on utilise le preprocessor classification
        return joblib.load(rul_path), None
    return None, None

@st.cache_data
def load_data():
    return pd.read_csv(os.path.join(BASE_DIR, 'data', 'predictive_maintenance_v3.csv'))

preprocessor, model = load_models()
rul_model, rul_preprocessor = load_rul_model()
df = load_data()

FEATURE_NAMES = ['vibration_rms', 'temperature_motor', 'current_phase_avg',
                 'pressure_level', 'rpm', 'hours_since_maintenance',
                 'ambient_temp', 'machine_type_CNC', 'machine_type_Compressor',
                 'machine_type_Pump', 'machine_type_Robotic Arm',
                 'operating_mode_idle', 'operating_mode_normal', 'operating_mode_peak']

NUMERIC_FEATURES = ['vibration_rms', 'temperature_motor', 'current_phase_avg',
                    'pressure_level', 'rpm', 'hours_since_maintenance', 'ambient_temp']

FEATURE_LABELS = {
    'vibration_rms': 'Vibration RMS', 'temperature_motor': 'Température moteur',
    'current_phase_avg': 'Courant de phase', 'pressure_level': 'Pression',
    'rpm': 'RPM', 'hours_since_maintenance': 'Heures depuis maintenance',
    'ambient_temp': 'Température ambiante'
}

def theme(fig, height=400):
    fig.update_layout(height=height, paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(240,248,240,0.4)', font=dict(color=C_DARK, family='Arial'),
        margin=dict(l=40, r=20, t=40, b=40),
        xaxis=dict(gridcolor=C_BORDER, linecolor=C_BORDER, tickfont=dict(color=C_GREY)),
        yaxis=dict(gridcolor=C_BORDER, linecolor=C_BORDER, tickfont=dict(color=C_GREY)),
        legend=dict(bgcolor='rgba(255,255,255,0.9)', font=dict(color=C_DARK), bordercolor=C_BORDER, borderwidth=1))
    return fig

def predict_via_api(api_url, payload):
    try:
        resp = requests.post(f"{api_url}/predict", json=payload, timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            return data["probability"], data["risk_level"], data["recommendation"], True
        return None, None, None, False
    except Exception:
        return None, None, None, False

def predict_local(preprocessor, model, input_data):
    proba = float(model.predict_proba(preprocessor.transform(input_data))[0][1])
    if proba >= 0.70:
        risk_level, recommendation = "ELEVE", "Intervention de maintenance recommandée dans les 24h"
    elif proba >= 0.30:
        risk_level, recommendation = "MODERE", "Surveillance renforcée recommandée"
    else:
        risk_level, recommendation = "FAIBLE", "Aucune action immédiate requise"
    return proba, risk_level, recommendation

def check_api_health(api_url):
    try:
        resp = requests.get(f"{api_url}/health", timeout=3)
        if resp.status_code == 200:
            return True, resp.json()
        return False, {}
    except Exception:
        return False, {}

def generate_parc_machines(preprocessor, model, rul_model, rul_preprocessor):
    np.random.seed(42)
    machine_types = ['CNC', 'Pump', 'Compressor', 'Robotic Arm']
    modes = ['normal', 'idle', 'peak']
    machines = []
    for i in range(1, 16):
        mt   = machine_types[i % 4]
        mode = modes[i % 3]
        vib  = round(np.random.uniform(0.5, 9.5), 2)
        temp = round(np.random.uniform(35, 90), 1)
        curr = round(np.random.uniform(2.2, 30.0), 1)
        pres = round(np.random.uniform(10.0, 200.0), 1)
        rpm  = round(np.random.uniform(124.0, 4000.0), 0)
        hrs  = round(np.random.uniform(0.0, 575.0), 0)
        amb  = round(np.random.uniform(8.0, 18.0), 1)

        input_df = pd.DataFrame([{
            'vibration_rms': vib, 'temperature_motor': temp,
            'current_phase_avg': curr, 'pressure_level': pres,
            'rpm': rpm, 'hours_since_maintenance': hrs,
            'ambient_temp': amb, 'machine_type': mt, 'operating_mode': mode
        }])

        # Prédiction panne avec le VRAI modèle XGBoost
        prob = float(model.predict_proba(preprocessor.transform(input_df))[0][1])

        # Prédiction RUL avec le VRAI modèle Random Forest
        # Le modèle RUL est un Pipeline complet (preprocessor + regressor)
        # Il prend les données BRUTES directement
        rul_val = max(0.0, float(rul_model.predict(input_df)[0])) if rul_model is not None else max(0, np.random.uniform(0, 98) * (1 - prob))

        machines.append({
            'ID': f'M{i:03d}', 'Type': mt, 'Mode': mode,
            'Vibration (mm/s)': vib, 'Temp. moteur (°C)': temp,
            'Prob. panne (%)': round(prob * 100, 1),
            'RUL estimé (h)': round(rul_val, 1),
            'Statut': '🔴 CRITIQUE' if prob >= 0.70 else ('🟠 ATTENTION' if prob >= 0.30 else '🟢 OK'),
        })
    return pd.DataFrame(machines).sort_values('Prob. panne (%)', ascending=False)

# ─── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    logo_path = os.path.join(BASE_DIR, 'assets', 'efrei_logo.png')
    if os.path.exists(logo_path):
        st.image(logo_path, width=150)

    rul_status = "✅ Chargé" if rul_model is not None else "❌ Non trouvé"
    rul_prep_status = "✅ Chargé" if rul_preprocessor is not None else "⚠️ Fallback classification"
    st.markdown(f"""
    <div style='padding:14px 0 20px 0; border-bottom:1px solid {C_BORDER}; margin-bottom:24px;'>
      <div style='font-size:17px; font-weight:800; color:{C_DARK};'>MaintPredict</div>
      <div style='font-size:11px; color:{C_GREY}; margin-top:3px; text-transform:uppercase; letter-spacing:1px;'>Système IA industriel — M1 DE 2025-26</div>
    </div>
    <div class='nav-label'>Navigation</div>
    """, unsafe_allow_html=True)

    page = st.radio("Navigation", [
        "Vue d'ensemble", "Analyse des données", "Prédiction temps réel",
        "Durée de vie restante (RUL)", "État du parc machines", "Impact économique",
        "Performance des modèles", "Interprétabilité SHAP"
    ], label_visibility="collapsed")

    st.markdown(f"""<div style='margin-top:24px; font-size:10px; color:{C_GREY}; text-transform:uppercase; letter-spacing:1.5px; font-weight:700; margin-bottom:6px;'>Configuration API</div>""", unsafe_allow_html=True)
    api_url = st.text_input("URL de l'API", value=DEFAULT_API_URL, label_visibility="collapsed")
    api_ok, api_info = check_api_health(api_url)

    if api_ok:
        st.markdown(f"""<div style='background:#F0FDF4; border:1px solid #BBF7D0; border-radius:8px; padding:10px 14px; font-size:12px; color:{C_GREEN}; margin-bottom:8px;'>✅ <b>API connectée</b><br>Modèle : {api_info.get('model', 'XGBoost')}</div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div style='background:#FEF2F2; border:1px solid #FECACA; border-radius:8px; padding:10px 14px; font-size:12px; color:{C_RED}; margin-bottom:8px;'>⚠️ <b>API non disponible</b><br>Mode : prédiction locale</div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div style='margin-top:24px; font-size:12px; color:{C_GREY}; border-top:1px solid {C_BORDER}; padding-top:16px; line-height:2;'>
      <div style='color:{C_GREY}; font-size:10px; text-transform:uppercase; letter-spacing:1.5px; font-weight:700; margin-bottom:6px;'>Modèle de prédiction</div>
      <b style='color:{C_GREEN}'>XGBoost</b> (modèle retenu)<br>
      ROC-AUC : <b style='color:{C_GREEN}'>0.9955</b><br>
      Recall : <b style='color:{C_GREEN}'>95.5%</b><br>
      Seuil décision : <b style='color:{C_ORANGE}'>0.70</b><br><br>
      RUL (Bonus) : <b style='color:{C_GREEN}'>Random Forest</b><br>
      MAE : <b style='color:{C_GREEN}'>9.42h</b> · R² : <b style='color:{C_GREEN}'>0.67</b>
    </div>
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — VUE D'ENSEMBLE (orientée métier)
# ═══════════════════════════════════════════════════════════════════════════════
if "Vue d'ensemble" in page:
    st.markdown(f"""<div class='page-header'>
      <h1>Tableau de bord — Maintenance Prédictive <span class='badge'>Live</span></h1>
      <p>Système IA de détection de pannes industrielles · Modèle XGBoost · {len(df):,} observations analysées</p>
    </div>""", unsafe_allow_html=True)

    # ── KPI métier principaux ──────────────────────────────────────────────────
    n_pannes    = int(df['failure_within_24h'].sum())
    n_saines    = len(df) - n_pannes
    taux        = df['failure_within_24h'].mean()
    df_parc     = generate_parc_machines(preprocessor, model, rul_model, rul_preprocessor)
    n_critique  = len(df_parc[df_parc['Statut'].str.contains('CRITIQUE')])
    n_attention = len(df_parc[df_parc['Statut'].str.contains('ATTENTION')])
    n_ok        = len(df_parc[df_parc['Statut'].str.contains('OK')])

    st.markdown("### 🏭 État actuel du parc machines")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""<div class='kpi-alert'>
          <div style='font-size:11px; font-weight:700; color:{C_RED}; text-transform:uppercase; letter-spacing:1px;'>🔴 Machines critiques</div>
          <div style='font-size:42px; font-weight:900; color:{C_RED}; line-height:1.2;'>{n_critique}</div>
          <div style='font-size:12px; color:{C_GREY};'>Intervention requise &lt; 24h</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""<div class='kpi-warn'>
          <div style='font-size:11px; font-weight:700; color:{C_ORANGE}; text-transform:uppercase; letter-spacing:1px;'>🟠 Machines en attention</div>
          <div style='font-size:42px; font-weight:900; color:{C_ORANGE}; line-height:1.2;'>{n_attention}</div>
          <div style='font-size:12px; color:{C_GREY};'>Inspection sous 48-72h</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""<div class='kpi-ok'>
          <div style='font-size:11px; font-weight:700; color:{C_GREEN}; text-transform:uppercase; letter-spacing:1px;'>🟢 Machines opérationnelles</div>
          <div style='font-size:42px; font-weight:900; color:{C_GREEN}; line-height:1.2;'>{n_ok}</div>
          <div style='font-size:12px; color:{C_GREY};'>Aucune action requise</div>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""<div style='background:{C_PANEL}; border:1px solid {C_BORDER}; border-top:4px solid {C_BLUE}; border-radius:10px; padding:20px; text-align:center;'>
          <div style='font-size:11px; font-weight:700; color:{C_BLUE}; text-transform:uppercase; letter-spacing:1px;'>📊 Total parc</div>
          <div style='font-size:42px; font-weight:900; color:{C_DARK}; line-height:1.2;'>{len(df_parc)}</div>
          <div style='font-size:12px; color:{C_GREY};'>machines surveillées</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Alertes prioritaires ───────────────────────────────────────────────────
    st.markdown("### 🚨 Alertes prioritaires — Machines à risque")
    top_alertes = df_parc[df_parc['Statut'].str.contains('CRITIQUE|ATTENTION')].head(5)
    if len(top_alertes) == 0:
        st.success("✅ Aucune machine en état critique actuellement.")
    else:
        cols = st.columns(min(len(top_alertes), 5))
        for i, (_, row) in enumerate(top_alertes.iterrows()):
            with cols[i]:
                is_crit = 'CRITIQUE' in row['Statut']
                color = C_RED if is_crit else C_ORANGE
                bg    = "#FEF2F2" if is_crit else "#FFFBEB"
                st.markdown(f"""<div style='background:{bg}; border:1px solid {color}; border-top:4px solid {color}; border-radius:10px; padding:16px; text-align:center;'>
                  <div style='font-size:18px; font-weight:900; color:{color};'>{row['ID']}</div>
                  <div style='font-size:11px; color:{C_GREY}; margin:3px 0;'>{row['Type']}</div>
                  <div style='font-size:28px; font-weight:800; color:{color};'>{row['Prob. panne (%)']:.0f}%</div>
                  <div style='font-size:10px; color:{C_GREY};'>risque panne</div>
                  <div style='margin-top:8px; font-size:12px; color:{color}; font-weight:600;'>RUL : {row['RUL estimé (h)']:.0f}h</div>
                  <div style='font-size:10px; color:{C_GREY};'>{row['Statut']}</div>
                </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Stats dataset historique + perf modèle ────────────────────────────────
    st.markdown("### 📈 Données historiques & performance du modèle IA")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Répartition des pannes — dataset historique (24 042 observations)**")
        counts = df['failure_within_24h'].value_counts()
        fig = go.Figure(go.Pie(
            values=[n_saines, n_pannes],
            labels=['Machine saine', 'Panne dans les 24h'],
            marker=dict(colors=[C_GREEN2, C_RED]),
            hole=0.55,
            textfont=dict(color=C_WHITE, size=13),
            textinfo='label+percent'
        ))
        fig.add_annotation(
            text=f"<b>{n_pannes:,}</b><br>pannes<br>détectées",
            x=0.5, y=0.5,
            font=dict(size=14, color=C_DARK),
            showarrow=False
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(font=dict(color=C_DARK), bgcolor='rgba(255,255,255,0.9)'),
            height=320, margin=dict(l=20,r=20,t=20,b=20)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"""<div class='info-card'>
          Sur <b>{len(df):,} enregistrements</b> historiques :<br>
          • <b style='color:{C_RED}'>{n_pannes:,} pannes</b> détectées ({taux:.1%} du parc)<br>
          • <b style='color:{C_GREEN}'>{n_saines:,} machines saines</b> ({1-taux:.1%})<br>
          Déséquilibre géré par <b>class_weight + seuil 0.70</b>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown("**Performance du modèle XGBoost — sélectionné pour la production**")
        metriques = {
            'ROC-AUC': 0.9955, 'Recall': 0.9551,
            'Précision': 0.8820, 'F1-Score': 0.9171, 'Accuracy': 0.9744
        }
        fig = go.Figure()
        colors_bar = [C_GREEN, C_GREEN, C_GREEN2, C_GREEN2, C_GREEN3]
        for (nom, val), col in zip(metriques.items(), colors_bar):
            fig.add_trace(go.Bar(
                x=[nom], y=[val],
                marker_color=col,
                text=[f"{val:.4f}"],
                textposition='outside',
                textfont=dict(color=C_DARK, size=13),
                name=nom
            ))
        fig.update_layout(
            showlegend=False,
            yaxis=dict(range=[0.85, 1.02], gridcolor=C_BORDER),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(240,248,240,0.4)',
            font=dict(color=C_DARK),
            height=320, margin=dict(l=20,r=20,t=20,b=40)
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(f"""<div class='info-card'>
          Modèle <b>XGBoost</b> retenu (meilleur parmi 4 testés) :<br>
          • Détecte <b style='color:{C_GREEN}'>95.5% des pannes</b> (Recall)<br>
          • <b style='color:{C_GREEN}'>1 fausse alarme sur 8</b> seulement (Précision 88%)<br>
          • Seuil de décision optimisé à <b>0.70</b> pour maximiser le F1
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Taux de panne par type de machine ────────────────────────────────────
    st.markdown("### 🔧 Taux de panne par type de machine et mode opératoire")
    col1, col2 = st.columns(2)
    with col1:
        ft = df.groupby('machine_type')['failure_within_24h'].agg(['sum','count','mean']).reset_index()
        ft.columns = ['Type', 'Nb pannes', 'Total', 'Taux']
        fig = px.bar(ft.sort_values('Taux', ascending=False), x='Type', y='Nb pannes',
            title="Nombre de pannes par type de machine",
            color='Taux', color_continuous_scale=[[0,C_GREEN3],[0.5,C_GREEN2],[1,C_RED]],
            text='Nb pannes')
        fig.update_traces(textposition='outside', textfont=dict(color=C_DARK))
        st.plotly_chart(theme(fig, 320), use_container_width=True)
    with col2:
        fm = df.groupby('operating_mode')['failure_within_24h'].agg(['sum','mean']).reset_index()
        fm.columns = ['Mode', 'Nb pannes', 'Taux']
        fig = px.bar(fm.sort_values('Taux', ascending=True), x='Taux', y='Mode', orientation='h',
            title="Taux de panne par mode opératoire",
            color='Taux', color_continuous_scale=[[0,C_GREEN3],[0.5,C_GREEN2],[1,C_RED]],
            text=fm.sort_values('Taux')['Taux'].map(lambda x: f"{x:.1%}"))
        fig.update_traces(textposition='outside', textfont=dict(color=C_DARK))
        st.plotly_chart(theme(fig, 320), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — ANALYSE DES DONNÉES
# ═══════════════════════════════════════════════════════════════════════════════
elif "Analyse" in page:
    st.markdown(f"""<div class='page-header'><h1>Analyse Exploratoire des Données</h1><p>Distributions, corrélations et comportements des capteurs industriels</p></div>""", unsafe_allow_html=True)
    feature = st.selectbox("Capteur à analyser", NUMERIC_FEATURES, format_func=lambda x: FEATURE_LABELS.get(x, x))
    col1, col2 = st.columns(2)
    with col1:
        fig = px.histogram(df, x=feature, color='failure_within_24h',
            color_discrete_map={0: C_GREEN2, 1: C_RED}, barmode='overlay', opacity=0.75,
            title=f"Distribution — {FEATURE_LABELS.get(feature, feature)}")
        fig.for_each_trace(lambda t: t.update(name='Sain' if t.name=='0' else 'Panne'))
        st.plotly_chart(theme(fig), use_container_width=True)
    with col2:
        fig = px.box(df, x='failure_within_24h', y=feature, color='failure_within_24h',
            color_discrete_map={0: C_GREEN2, 1: C_RED}, title="Boxplot — OK vs Panne")
        fig.for_each_trace(lambda t: t.update(name='Sain' if t.name=='0' else 'Panne'))
        st.plotly_chart(theme(fig), use_container_width=True)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        corr = df[NUMERIC_FEATURES + ['failure_within_24h']].corr()
        fig = px.imshow(corr, text_auto='.2f', color_continuous_scale=[[0,C_BLUE],[0.5,'#FFFFFF'],[1,C_GREEN]],
            title="Matrice de corrélation des capteurs")
        st.plotly_chart(theme(fig, 450), use_container_width=True)
    with col2:
        st.markdown("<h3>Statistiques descriptives</h3>", unsafe_allow_html=True)
        st.dataframe(df[NUMERIC_FEATURES].describe().round(3), use_container_width=True, height=420)
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.markdown(f"<div class='info-card'>Ratio déséquilibre : <b>5.8:1</b><br>85.2% sain / 14.8% panne</div>", unsafe_allow_html=True)
    c2.markdown(f"<div class='info-card'>Accuracy seule <b>insuffisante</b> — modèle naïf = 85.2% sans détecter une seule panne</div>", unsafe_allow_html=True)
    c3.markdown(f"<div class='info-card'>Métriques retenues : <b>Recall, F1-score, ROC-AUC, PR-AUC</b></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — PRÉDICTION PANNE
# ═══════════════════════════════════════════════════════════════════════════════
elif "Prédiction temps" in page:
    api_badge = '<span class="api-badge">Via API</span>' if api_ok else f'<span style="font-size:11px;color:#E67E22;">⚠ Mode local</span>'
    st.markdown(f"""<div class='page-header'><h1>Prédiction de Panne en Temps Réel {api_badge}</h1><p>Saisissez les valeurs des capteurs pour évaluer le risque de panne dans les 24h — Modèle XGBoost</p></div>""", unsafe_allow_html=True)
    if api_ok:
        st.markdown(f"""<div style='background:#EFF6FF; border:1px solid #BFDBFE; border-radius:8px; padding:10px 16px; font-size:13px; color:{C_BLUE}; margin-bottom:16px;'>🔗 Prédictions transmises à l'API FastAPI — <code>{api_url}/predict</code></div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""<div style='background:#FFFBEB; border:1px solid #FDE68A; border-radius:8px; padding:10px 16px; font-size:13px; color:{C_ORANGE}; margin-bottom:16px;'>⚠️ API non disponible — prédiction locale via XGBoost. Lancez <code>uvicorn api.main:app --reload --port 8000</code></div>""", unsafe_allow_html=True)
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
        payload = {"machine_type": machine_type, "operating_mode": operating_mode,
            "vibration_rms": float(vibration), "temperature_motor": float(temperature),
            "current_phase_avg": float(current), "pressure_level": float(pressure),
            "rpm": float(rpm), "hours_since_maintenance": float(hours_maintenance), "ambient_temp": float(ambient)}
        if api_ok:
            proba, risk_level, recommendation, success = predict_via_api(api_url, payload)
            source_label = "🔗 Résultat via API FastAPI (XGBoost)"
            if not success:
                input_data = pd.DataFrame([payload])
                proba, risk_level, recommendation = predict_local(preprocessor, model, input_data)
                source_label = "⚠️ Résultat local XGBoost (erreur API)"
        else:
            input_data = pd.DataFrame([payload])
            proba, risk_level, recommendation = predict_local(preprocessor, model, input_data)
            source_label = "💾 Résultat local XGBoost (API indisponible)"
        prediction = 1 if proba >= 0.70 else 0
        st.markdown(f"<div style='font-size:12px; color:{C_GREY}; margin-bottom:12px;'>{source_label}</div>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col1:
            color = C_RED if prediction == 1 else C_GREEN
            bg    = "#FEF2F2" if prediction == 1 else "#F0FDF4"
            label = "⚠️ Risque de panne détecté" if prediction == 1 else "✅ Machine en bon état"
            rec_bg = "#FEF3C7" if prediction == 1 else "#DCFCE7"
            st.markdown(f"""<div style='background:{bg}; border:1px solid; border-left:4px solid {color}; border-radius:10px; padding:28px; text-align:center;'>
              <div style='font-size:13px; font-weight:700; color:{color}; text-transform:uppercase; letter-spacing:2px; margin-bottom:12px;'>{label}</div>
              <div style='font-size:48px; font-weight:900; color:{color}; line-height:1;'>{proba:.1%}</div>
              <div style='color:{C_GREY}; font-size:13px; margin-top:10px;'>Probabilité de panne dans les 24h</div>
              <div style='margin-top:20px; background:{rec_bg}; border-radius:8px; padding:12px; color:{color}; font-size:13px; font-weight:600;'>{recommendation}</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            bar_color = C_RED if proba >= 0.70 else (C_ORANGE if proba >= 0.30 else C_GREEN2)
            fig = go.Figure(go.Indicator(mode="gauge+number", value=proba*100,
                number=dict(suffix="%", font=dict(color=C_DARK, size=36)),
                title=dict(text="Indice de risque", font=dict(color=C_GREY, size=13)),
                gauge=dict(axis=dict(range=[0,100], tickfont=dict(color=C_GREY)),
                    bar=dict(color=bar_color, thickness=0.3), bgcolor=C_BG,
                    borderwidth=1, bordercolor=C_BORDER,
                    steps=[dict(range=[0,30], color='rgba(39,174,96,0.10)'),
                           dict(range=[30,70], color='rgba(230,126,34,0.10)'),
                           dict(range=[70,100], color='rgba(231,76,60,0.10)')],
                    threshold=dict(line=dict(color=C_ORANGE, width=3), value=70))))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color=C_DARK), height=280, margin=dict(l=20,r=20,t=40,b=20))
            st.plotly_chart(fig, use_container_width=True)
        if api_ok:
            with st.expander("🔍 Détail de la requête API"):
                col_req, col_resp = st.columns(2)
                with col_req:
                    st.markdown("**Requête envoyée (JSON)**"); st.json(payload)
                with col_resp:
                    st.markdown("**Réponse reçue**")
                    st.json({"prediction": prediction, "probability": round(proba, 4), "risk_level": risk_level, "recommendation": recommendation})
        st.markdown("---")
        st.markdown("<h3>Paramètres saisis</h3>", unsafe_allow_html=True)
        st.dataframe(pd.DataFrame([payload]).rename(columns=FEATURE_LABELS), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4 — DURÉE DE VIE RESTANTE (RUL) — TÂCHE BONUS
# ═══════════════════════════════════════════════════════════════════════════════
elif "RUL" in page:
    st.markdown(f"""<div class='page-header'><h1>Estimation de la Durée de Vie Restante <span class='badge'>Bonus</span></h1><p>Tâche prédictive bonus — Régression sur rul_hours · Modèle : Random Forest (MAE = 9.42h · R² = 0.67)</p></div>""", unsafe_allow_html=True)
    if rul_model is None:
        st.error("⚠️ Modèle RUL non trouvé. Lancez d'abord notebooks/05_Regression_RUL.ipynb pour entraîner models/rul_model.joblib")
    else:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Modèle retenu", "Random Forest")
        c2.metric("MAE", "9.42 heures")
        c3.metric("RMSE", "15.11 heures")
        c4.metric("R²", "0.67")
        st.markdown("---")
        st.markdown("<h3>Comparaison des 4 modèles — Régression RUL</h3>", unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        rul_results = {
            'Random Forest':       {'MAE': 9.42,  'RMSE': 15.11, 'R²': 0.67, 'MAE CV': 9.51},
            'XGBoost':             {'MAE': 10.68, 'RMSE': 15.56, 'R²': 0.65, 'MAE CV': 10.86},
            'MLP (Deep Learning)': {'MAE': 11.65, 'RMSE': 16.40, 'R²': 0.61, 'MAE CV': 11.94},
            'Ridge (baseline)':    {'MAE': 20.47, 'RMSE': 24.39, 'R²': 0.14, 'MAE CV': 20.73},
        }
        df_rul = pd.DataFrame(rul_results).T.reset_index()
        df_rul.columns = ['Modèle', 'MAE (h)', 'RMSE (h)', 'R²', 'MAE CV (h)']
        with col1:
            fig = px.bar(df_rul.sort_values('MAE (h)', ascending=True), x='MAE (h)', y='Modèle', orientation='h',
                title="MAE par modèle — plus bas = mieux", color='MAE (h)',
                color_continuous_scale=[[0,C_GREEN],[0.5,C_ORANGE],[1,C_RED]],
                text=df_rul.sort_values('MAE (h)', ascending=True)['MAE (h)'].map(lambda x: f"{x:.2f}h"))
            fig.update_traces(textposition='outside', textfont=dict(color=C_DARK))
            st.plotly_chart(theme(fig, 320), use_container_width=True)
        with col2:
            fig = px.bar(df_rul.sort_values('R²', ascending=True), x='R²', y='Modèle', orientation='h',
                title="R² par modèle — plus haut = mieux", color='R²',
                color_continuous_scale=[[0,C_RED],[0.5,C_ORANGE],[1,C_GREEN]],
                text=df_rul.sort_values('R²', ascending=True)['R²'].map(lambda x: f"{x:.2f}"))
            fig.update_traces(textposition='outside', textfont=dict(color=C_DARK))
            st.plotly_chart(theme(fig, 320), use_container_width=True)
        st.dataframe(df_rul.style
            .highlight_min(axis=0, subset=['MAE (h)','RMSE (h)','MAE CV (h)'], color='rgba(27,138,62,0.15)')
            .highlight_max(axis=0, subset=['R²'], color='rgba(27,138,62,0.15)')
            .format({'MAE (h)':'{:.2f}','RMSE (h)':'{:.2f}','R²':'{:.2f}','MAE CV (h)':'{:.2f}'}), use_container_width=True)
        st.markdown("---")
        st.markdown("<h3>Simulateur — Estimer la durée de vie restante</h3>", unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            rul_machine = st.selectbox("Type de machine", ['CNC','Pump','Compressor','Robotic Arm'], key='rul_machine')
            rul_mode    = st.selectbox("Mode opératoire", ['normal','idle','peak'], key='rul_mode')
            rul_vib     = st.number_input("Vibration RMS (mm/s)", min_value=0.35, max_value=10.0, value=2.0, step=0.05, key='rul_vib')
        with col2:
            rul_temp     = st.number_input("Température moteur (°C)", min_value=28.0, max_value=95.0, value=55.0, step=0.5, key='rul_temp')
            rul_current  = st.number_input("Courant phase moyen (A)", min_value=2.2, max_value=35.0, value=9.0, step=0.1, key='rul_current')
            rul_pressure = st.number_input("Pression (bar)", min_value=10.0, max_value=206.0, value=60.0, step=1.0, key='rul_pressure')
        with col3:
            rul_rpm   = st.number_input("RPM", min_value=124.0, max_value=4098.0, value=1200.0, step=1.0, key='rul_rpm')
            rul_hours = st.number_input("Heures depuis maintenance", min_value=0.0, max_value=575.0, value=150.0, step=1.0, key='rul_hours')
            rul_amb   = st.number_input("Température ambiante (°C)", min_value=8.0, max_value=18.0, value=13.0, step=0.5, key='rul_amb')
        if st.button("Estimer la durée de vie restante", type="primary", key='btn_rul'):
            rul_input = pd.DataFrame([{
                'vibration_rms': float(rul_vib), 'temperature_motor': float(rul_temp),
                'current_phase_avg': float(rul_current), 'pressure_level': float(rul_pressure),
                'rpm': float(rul_rpm), 'hours_since_maintenance': float(rul_hours),
                'ambient_temp': float(rul_amb), 'machine_type': rul_machine, 'operating_mode': rul_mode
            }])
            # Utilise le preprocessor RUL dédié, ou fallback sur le preprocessor classification
            active_rul_preprocessor = rul_preprocessor if rul_preprocessor is not None else preprocessor
            rul_input_processed = active_rul_preprocessor.transform(rul_input)
            # Le modèle RUL est un Pipeline complet → prend les données brutes
            rul_pred = max(0, float(rul_model.predict(rul_input)[0]))
            urgency_color = C_RED if rul_pred < 10 else (C_ORANGE if rul_pred < 24 else C_GREEN)
            urgency_label = "INTERVENTION URGENTE" if rul_pred < 10 else ("PLANIFIER MAINTENANCE" if rul_pred < 24 else "MACHINE OPÉRATIONNELLE")
            urgency_bg    = "#FEF2F2" if rul_pred < 10 else ("#FFFBEB" if rul_pred < 24 else "#F0FDF4")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.markdown(f"""<div style='background:{urgency_bg}; border-left:4px solid {urgency_color}; border-radius:10px; padding:28px; text-align:center;'>
                  <div style='font-size:13px; font-weight:700; color:{urgency_color}; text-transform:uppercase; letter-spacing:2px; margin-bottom:12px;'>{urgency_label}</div>
                  <div style='font-size:56px; font-weight:900; color:{urgency_color}; line-height:1;'>{rul_pred:.0f}h</div>
                  <div style='color:{C_GREY}; font-size:13px; margin-top:10px;'>Durée de vie restante estimée</div>
                  <div style='margin-top:8px; color:{C_GREY}; font-size:12px;'>Précision : ±9.42 heures (MAE)</div>
                </div>""", unsafe_allow_html=True)
            with col2:
                fig = go.Figure(go.Indicator(mode="gauge+number", value=min(rul_pred, 98),
                    number=dict(suffix="h", font=dict(color=C_DARK, size=36)),
                    title=dict(text="Heures restantes estimées", font=dict(color=C_GREY, size=13)),
                    gauge=dict(axis=dict(range=[0,98], tickfont=dict(color=C_GREY)),
                        bar=dict(color=urgency_color, thickness=0.3), bgcolor=C_BG,
                        borderwidth=1, bordercolor=C_BORDER,
                        steps=[dict(range=[0,10], color='rgba(231,76,60,0.10)'),
                               dict(range=[10,24], color='rgba(230,126,34,0.10)'),
                               dict(range=[24,98], color='rgba(39,174,96,0.10)')],
                        threshold=dict(line=dict(color=C_ORANGE, width=3), value=24))))
                fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(color=C_DARK), height=280, margin=dict(l=20,r=20,t=40,b=20))
                st.plotly_chart(fig, use_container_width=True)
        st.markdown("---")
        st.markdown(f"""<div class='info-card'><b>Interprétation métier :</b> Le modèle Random Forest estime la durée de vie restante avec une erreur moyenne de <b>±9.42 heures</b>. Un responsable maintenance peut planifier ses interventions avec une fenêtre de confiance de ±9h.<br><br><b>Seuils :</b> &lt;10h = intervention urgente · 10–24h = planifier maintenance · &gt;24h = opérationnel</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5 — ÉTAT DU PARC MACHINES
# ═══════════════════════════════════════════════════════════════════════════════
elif "parc" in page:
    st.markdown(f"""<div class='page-header'><h1>État du Parc Machines <span class='badge'>Temps réel</span></h1><p>Vue d'ensemble des 15 machines — Statut, risque de panne et durée de vie restante</p></div>""", unsafe_allow_html=True)
    df_parc = generate_parc_machines(preprocessor, model, rul_model, rul_preprocessor)
    n_critique  = len(df_parc[df_parc['Statut'].str.contains('CRITIQUE')])
    n_attention = len(df_parc[df_parc['Statut'].str.contains('ATTENTION')])
    n_ok        = len(df_parc[df_parc['Statut'].str.contains('OK')])
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total machines", len(df_parc))
    c2.metric("🔴 Critiques", n_critique, f"{n_critique/len(df_parc):.0%} du parc")
    c3.metric("🟠 Attention", n_attention)
    c4.metric("🟢 Opérationnelles", n_ok)
    st.markdown("---")
    st.markdown("<h3>🚨 Top alertes — Intervention prioritaire</h3>", unsafe_allow_html=True)
    top3 = df_parc[df_parc['Statut'].str.contains('CRITIQUE')].head(3)
    if len(top3) == 0:
        top3 = df_parc.head(3)
    cols = st.columns(len(top3))
    for i, (_, row) in enumerate(top3.iterrows()):
        with cols[i]:
            color = C_RED if 'CRITIQUE' in row['Statut'] else C_ORANGE
            bg    = "#FEF2F2" if 'CRITIQUE' in row['Statut'] else "#FFFBEB"
            st.markdown(f"""<div style='background:{bg}; border:1px solid {color}; border-top:4px solid {color}; border-radius:10px; padding:18px; text-align:center;'>
              <div style='font-size:20px; font-weight:900; color:{color};'>{row['ID']}</div>
              <div style='font-size:12px; color:{C_GREY}; margin:4px 0;'>{row['Type']} · {row['Mode']}</div>
              <div style='font-size:28px; font-weight:800; color:{color};'>{row['Prob. panne (%)']:.0f}%</div>
              <div style='font-size:11px; color:{C_GREY};'>Risque de panne</div>
              <div style='margin-top:10px; font-size:13px; color:{color}; font-weight:600;'>RUL : {row['RUL estimé (h)']:.0f}h</div>
            </div>""", unsafe_allow_html=True)
    st.markdown("---")
    col1, col2 = st.columns([1.5, 1])
    with col1:
        st.markdown("<h3>État détaillé de toutes les machines</h3>", unsafe_allow_html=True)
        st.dataframe(df_parc.style.apply(
            lambda row: ['background-color: #FEF2F2' if 'CRITIQUE' in str(row.get('Statut',''))
                         else ('background-color: #FFFBEB' if 'ATTENTION' in str(row.get('Statut',''))
                         else 'background-color: #F0FDF4')] * len(row), axis=1),
            use_container_width=True, height=480)
    with col2:
        statut_counts = df_parc['Statut'].apply(
            lambda x: 'Critique' if 'CRITIQUE' in x else ('Attention' if 'ATTENTION' in x else 'OK')
        ).value_counts()
        fig = go.Figure(go.Pie(values=statut_counts.values, labels=statut_counts.index,
            marker=dict(colors=[C_RED, C_ORANGE, C_GREEN2]), hole=0.55, textfont=dict(color=C_WHITE, size=13)))
        fig.update_layout(title=dict(text="Répartition des statuts", font=dict(color=C_DARK)),
            paper_bgcolor='rgba(0,0,0,0)', legend=dict(font=dict(color=C_DARK), bgcolor='rgba(255,255,255,0.9)'),
            height=280, margin=dict(l=20,r=20,t=40,b=20))
        st.plotly_chart(fig, use_container_width=True)
        fig2 = px.scatter(df_parc, x='RUL estimé (h)', y='Prob. panne (%)', color='Type',
            hover_data=['ID','Type','Mode'], title="Risque vs Durée de vie restante",
            color_discrete_sequence=[C_BLUE, C_GREEN, C_ORANGE, C_RED])
        fig2.add_hline(y=70, line_dash="dash", line_color=C_RED, annotation_text="Seuil 70%")
        fig2.add_vline(x=24, line_dash="dash", line_color=C_ORANGE, annotation_text="24h")
        st.plotly_chart(theme(fig2, 280), use_container_width=True)
    st.markdown("---")
    st.markdown(f"""<div class='info-card'><b>Plan d'intervention recommandé :</b><br>🔴 <b>{n_critique} machine(s) critique(s)</b> — intervention dans les 24h<br>🟠 <b>{n_attention} machine(s) en attention</b> — planifier inspection sous 48-72h<br>🟢 <b>{n_ok} machine(s) opérationnelle(s)</b> — maintenance préventive selon planning</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6 — IMPACT ÉCONOMIQUE
# ═══════════════════════════════════════════════════════════════════════════════
elif "économique" in page:
    st.markdown(f"""<div class='page-header'><h1>Impact Économique — ROI du Système IA</h1><p>Comparaison des coûts avec et sans système de maintenance prédictive</p></div>""", unsafe_allow_html=True)
    st.markdown("<h3>⚙️ Paramètres de simulation</h3>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        n_machines    = st.slider("Nombre de machines dans le parc", 5, 100, 20)
        taux_panne_an = st.slider("Pannes non planifiées par machine / an (sans IA)", 1, 10, 4)
    with col2:
        cout_panne  = st.number_input("Coût moyen d'une panne non planifiée (€)", 5000, 100000, 15000, step=1000)
        cout_fausse = st.number_input("Coût d'une fausse alarme (€)", 100, 2000, 500, step=100)
    with col3:
        recall_modele = st.slider("Recall du modèle (%)", 50, 100, 96)
        precision_mod = st.slider("Précision du modèle (%)", 50, 100, 88)
    st.markdown("---")
    recall = recall_modele / 100
    precision = precision_mod / 100
    total_pannes_an      = n_machines * taux_panne_an
    pannes_detectees     = int(total_pannes_an * recall)
    pannes_non_detectees = total_pannes_an - pannes_detectees
    fausses_alarmes      = int(pannes_detectees * (1 - precision) / precision)
    cout_sans_ia         = total_pannes_an * cout_panne
    cout_pannes_manquees = pannes_non_detectees * cout_panne
    cout_fausses_alarmes_total = fausses_alarmes * cout_fausse
    cout_interventions_ia= pannes_detectees * COUT_INTERVENTION_IA
    cout_avec_ia         = cout_pannes_manquees + cout_fausses_alarmes_total + cout_interventions_ia
    economies            = cout_sans_ia - cout_avec_ia
    roi_pct              = (economies / cout_sans_ia * 100) if cout_sans_ia > 0 else 0
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Coût sans IA / an", f"{cout_sans_ia:,.0f} €")
    c2.metric("Coût avec IA / an", f"{cout_avec_ia:,.0f} €", f"-{economies:,.0f} €")
    c3.metric("Économies annuelles", f"{economies:,.0f} €", f"ROI = {roi_pct:.0f}%")
    c4.metric("Pannes détectées", f"{pannes_detectees}/{total_pannes_an}", f"Recall {recall_modele}%")
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        fig = go.Figure()
        for cat, val, color in zip(['Sans IA','Avec IA'], [cout_sans_ia, cout_avec_ia], [C_RED, C_GREEN]):
            fig.add_trace(go.Bar(x=[cat], y=[val], marker_color=color,
                text=[f"{val:,.0f} €"], textposition='outside', textfont=dict(color=C_DARK, size=14), name=cat))
        fig.update_layout(title="Comparaison des coûts annuels totaux", showlegend=False,
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(240,248,240,0.4)', font=dict(color=C_DARK),
            height=380, margin=dict(l=40,r=20,t=50,b=40))
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = go.Figure(go.Pie(
            values=[cout_pannes_manquees, cout_fausses_alarmes_total, cout_interventions_ia],
            labels=['Pannes non détectées','Fausses alarmes','Interventions planifiées'],
            marker=dict(colors=[C_RED, C_ORANGE, C_GREEN2]), hole=0.4, textfont=dict(color=C_WHITE, size=12)))
        fig.update_layout(title="Décomposition du coût avec IA", paper_bgcolor='rgba(0,0,0,0)',
            legend=dict(font=dict(color=C_DARK), bgcolor='rgba(255,255,255,0.9)'),
            height=380, margin=dict(l=20,r=20,t=50,b=20))
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    annees = ['Année 1', 'Année 2', 'Année 3']
    economies_3ans = [economies, economies*1.08, economies*1.15]
    cout_ia_3ans   = [cout_avec_ia, cout_avec_ia*0.95, cout_avec_ia*0.92]
    cout_sans_3ans = [cout_sans_ia, cout_sans_ia*1.05, cout_sans_ia*1.10]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=annees, y=cout_sans_3ans, name='Sans IA',
        line=dict(color=C_RED, width=3), mode='lines+markers+text',
        text=[f"{v:,.0f}€" for v in cout_sans_3ans], textposition='top center'))
    fig.add_trace(go.Scatter(x=annees, y=cout_ia_3ans, name='Avec IA',
        line=dict(color=C_GREEN, width=3), mode='lines+markers+text',
        text=[f"{v:,.0f}€" for v in cout_ia_3ans], textposition='bottom center',
        fill='tonexty', fillcolor='rgba(27,138,62,0.08)'))
    fig.update_layout(title="Évolution des coûts sur 3 ans — Zone verte = économies réalisées",
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(240,248,240,0.4)', font=dict(color=C_DARK),
        legend=dict(bgcolor='rgba(255,255,255,0.9)', font=dict(color=C_DARK)),
        height=350, margin=dict(l=40,r=20,t=50,b=40))
    st.plotly_chart(fig, use_container_width=True)
    total_3ans = sum(economies_3ans)
    st.markdown(f"""<div class='info-card'><b>Synthèse ROI sur 3 ans :</b><br>Économies cumulées estimées : <b>{total_3ans:,.0f} €</b> sur {n_machines} machines<br>Réduction de <b>{roi_pct:.0f}%</b> des coûts de maintenance annuels.</div>""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7 — PERFORMANCE DES MODÈLES
# ═══════════════════════════════════════════════════════════════════════════════
elif "Performance" in page:
    st.markdown(f"""<div class='page-header'><h1>Comparaison des Modèles ML et DL</h1><p>Évaluation sur le jeu de test — 4 809 observations (20%) · Modèle retenu en production : XGBoost</p></div>""", unsafe_allow_html=True)
    results = {
        'XGBoost':             {'Accuracy':0.9744,'Precision':0.8820,'Recall':0.9551,'F1':0.9171,'ROC-AUC':0.9955,'PR-AUC':0.9741},
        'Random Forest':       {'Accuracy':0.9682,'Precision':0.9010,'Recall':0.8820,'F1':0.8914,'ROC-AUC':0.9938,'PR-AUC':0.9645},
        'MLP (Deep Learning)': {'Accuracy':0.9310,'Precision':0.7400,'Recall':0.8600,'F1':0.7900,'ROC-AUC':0.9739,'PR-AUC':0.8647},
        'Logistic Regression': {'Accuracy':0.9102,'Precision':0.6408,'Recall':0.8947,'F1':0.7468,'ROC-AUC':0.9588,'PR-AUC':0.8843},
    }
    cats = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'PR-AUC']
    df_res = pd.DataFrame(results).T.reset_index()
    df_res.columns = ['Modèle'] + cats
    colors_r = [C_GREEN, C_BLUE, C_ORANGE, C_GREY]
    cats_radar = ['Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC']
    fig = go.Figure()
    for (name, row), color in zip(results.items(), colors_r):
        vals = [row[c] for c in cats_radar]
        fig.add_trace(go.Scatterpolar(r=vals+[vals[0]], theta=cats_radar+[cats_radar[0]],
            fill='toself', line=dict(color=color, width=2), name=name, fillcolor='rgba(0,0,0,0.05)'))
    fig.update_layout(polar=dict(bgcolor='rgba(240,248,240,0.4)',
        radialaxis=dict(visible=True, range=[0.6,1], gridcolor=C_BORDER, tickfont=dict(color=C_GREY)),
        angularaxis=dict(gridcolor=C_BORDER, tickfont=dict(color=C_DARK))),
        paper_bgcolor='rgba(0,0,0,0)', font=dict(color=C_DARK),
        title=dict(text="Radar — Comparaison globale des 4 modèles", font=dict(color=C_DARK)),
        legend=dict(bgcolor='rgba(255,255,255,0.9)', font=dict(color=C_DARK), bordercolor=C_BORDER, borderwidth=1), height=420)
    st.plotly_chart(fig, use_container_width=True)
    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        metric = st.selectbox("Métrique à visualiser", cats)
        fig = px.bar(df_res.sort_values(metric, ascending=True), x=metric, y='Modèle', orientation='h',
            title=f"Comparaison — {metric}", color=metric,
            color_continuous_scale=[[0,C_GREEN3],[0.7,C_GREEN2],[1,C_GREEN]],
            text=df_res.sort_values(metric, ascending=True)[metric].map(lambda x: f"{x:.4f}"))
        fig.update_traces(textposition='outside', textfont=dict(color=C_DARK))
        st.plotly_chart(theme(fig, 320), use_container_width=True)
    with col2:
        st.dataframe(df_res.style.highlight_max(axis=0, subset=cats, color='rgba(27,138,62,0.15)').format({c:'{:.4f}' for c in cats}), use_container_width=True, height=220)
        st.markdown(f"""<div class='info-card' style='margin-top:12px;'><b>✅ Modèle retenu en production : XGBoost</b><br>
        Meilleur ROC-AUC (0.9955), PR-AUC (0.9741), Recall (0.9551) et F1 (0.9171).<br>
        Stable en validation croisée (std Recall = ±0.014).<br>
        60× moins de CO₂ que le MLP · Inférence &lt;1ms.<br>
        Seuil de décision optimisé à <b>0.70</b> pour maximiser le F1.</div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("<h3>Validation croisée — Stratified K-Fold (5 folds)</h3>", unsafe_allow_html=True)
    cv = pd.DataFrame({'Modèle':['XGBoost','Random Forest','Logistic Regression'],
        'Recall moyen':[0.9466,0.8683,0.8838], 'Recall std':['±0.0135','±0.0113','±0.0128'],
        'F1 moyen':[0.9127,0.8932,0.7454], 'ROC-AUC moyen':[0.9949,0.9930,0.9562]})
    st.dataframe(cv, use_container_width=True)
    st.markdown("---")
    st.markdown("<h3>Techniques de gestion du déséquilibre</h3>", unsafe_allow_html=True)
    imb = pd.DataFrame({'Technique':['class_weight (retenu)','Random Over-Sampling','SMOTE','Random Under-Sampling'],
        'Recall':[0.9551,0.9494,0.9438,0.9719], 'F1':[0.9171,0.9111,0.8924,0.8491],
        'ROC-AUC':[0.9955,0.9960,0.9949,0.9910], 'PR-AUC':[0.9741,0.9742,0.9794,0.9500]})
    fig = px.bar(imb, x='Technique', y=['Recall','F1','ROC-AUC','PR-AUC'], barmode='group',
        title="Comparaison des techniques de rééquilibrage",
        color_discrete_map={'Recall':C_RED,'F1':C_GREEN,'ROC-AUC':C_BLUE,'PR-AUC':C_ORANGE})
    st.plotly_chart(theme(fig, 380), use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 8 — INTERPRÉTABILITÉ SHAP
# ═══════════════════════════════════════════════════════════════════════════════
elif "SHAP" in page:
    st.markdown(f"""<div class='page-header'><h1>Interprétabilité — Modèle XGBoost</h1><p>Comprendre pourquoi le modèle déclenche une alerte de panne</p></div>""", unsafe_allow_html=True)
    importances = model.feature_importances_
    df_imp = pd.DataFrame({'Variable': FEATURE_NAMES, 'Importance': importances}).sort_values('Importance', ascending=True)
    fig = px.bar(df_imp, x='Importance', y='Variable', orientation='h',
        title="Feature Importance — XGBoost (réduction d'impureté Gini)", color='Importance',
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