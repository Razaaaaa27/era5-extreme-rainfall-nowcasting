import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title='Prediksi Hujan Ekstrem Aceh',
    page_icon=':cloud:',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Resolve paths relative to this script's own location, so it works
# regardless of the working directory Streamlit Cloud starts from
# (repo structure: repo_root/apps/app.py, repo_root/models/, repo_root/data/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, '..', 'models')
DATA_DIR = os.path.join(BASE_DIR, '..', 'data')

# Load models
@st.cache_resource
def load_all_models():
    """Load XGBoost and LSTM models from all folds"""
    models = {}
    for fold in [1, 2, 3]:
        with open(os.path.join(MODELS_DIR, f'xgb_model_fold{fold}.pkl'), 'rb') as f:
            models[f'xgb_fold{fold}'] = pickle.load(f)
        with open(os.path.join(MODELS_DIR, f'lstm_model_fold{fold}.pkl'), 'rb') as f:
            models[f'lstm_fold{fold}'] = pickle.load(f)
    return models

@st.cache_resource
def load_model_results():
    with open(os.path.join(MODELS_DIR, 'trained_models_results.pkl'), 'rb') as f:
        model_results = pickle.load(f)

    with open(os.path.join(MODELS_DIR, 'preprocessed_data_folds.pkl'), 'rb') as f:
        preprocessed_folds = pickle.load(f)

    with open(os.path.join(MODELS_DIR, 'model_interpretation_results.pkl'), 'rb') as f:
        interpretation = pickle.load(f)

    return model_results, preprocessed_folds, interpretation

@st.cache_data
def load_predictions():
    predictions = pd.read_csv(os.path.join(DATA_DIR, '2024', 'fold3_predictions.csv'))
    return predictions

# Calculate mean and std from lists
def calc_metrics_stats(scores_dict):
    """Calculate mean and std from list of fold scores"""
    stats = {}
    for metric, values in scores_dict.items():
        stats[f'{metric}_mean'] = np.mean(values)
        stats[f'{metric}_std'] = np.std(values)
    return stats

# Generate all 81 features from basic inputs
def generate_features_from_basic_input(basic_input):
    """Generate all 81 features from 8 basic meteorological inputs"""
    # Basic current values
    tp = basic_input['tp']
    ro = basic_input['ro']
    t2m = basic_input['t2m']
    u10 = basic_input['u10']
    v10 = basic_input['v10']
    swvl1 = basic_input['swvl1']
    wind_speed = basic_input['wind_speed']

    # For manual input, use reasonable defaults for lag features
    # Assume similar conditions in recent past (conservative approach)
    features = {
        # Current values (NOTE: no 'ro' in current, only lags)
        'tp': tp,
        't2m': t2m,
        'u10': u10,
        'v10': v10,
        'swvl1': swvl1,
        'wind_speed': wind_speed,

        # Lag features - tp
        'tp_lag1': tp * 0.8,  # Slightly lower in past
        'tp_lag2': tp * 0.6,
        'tp_lag3': tp * 0.5,
        'tp_lag6': tp * 0.3,
        'tp_lag7': tp * 0.25,
        'tp_lag14': tp * 0.1,

        # Lag features - ro
        'ro_lag1': ro * 0.9,
        'ro_lag2': ro * 0.8,
        'ro_lag3': ro * 0.7,
        'ro_lag6': ro * 0.5,
        'ro_lag7': ro * 0.4,
        'ro_lag14': ro * 0.2,

        # Lag features - t2m
        't2m_lag1': t2m,
        't2m_lag2': t2m,
        't2m_lag3': t2m,
        't2m_lag6': t2m - 0.5,
        't2m_lag7': t2m - 0.5,
        't2m_lag14': t2m - 1.0,

        # Lag features - u10
        'u10_lag1': u10,
        'u10_lag2': u10,
        'u10_lag3': u10,
        'u10_lag6': u10 * 0.8,
        'u10_lag7': u10 * 0.7,
        'u10_lag14': u10 * 0.5,

        # Lag features - v10
        'v10_lag1': v10,
        'v10_lag2': v10,
        'v10_lag3': v10,
        'v10_lag6': v10 * 0.8,
        'v10_lag7': v10 * 0.7,
        'v10_lag14': v10 * 0.5,

        # Lag features - swvl1
        'swvl1_lag1': swvl1,
        'swvl1_lag2': swvl1,
        'swvl1_lag3': swvl1 * 0.98,
        'swvl1_lag6': swvl1 * 0.95,
        'swvl1_lag7': swvl1 * 0.93,
        'swvl1_lag14': swvl1 * 0.85,

        # Lag features - wind_speed
        'wind_speed_lag1': wind_speed,
        'wind_speed_lag2': wind_speed,
        'wind_speed_lag3': wind_speed * 0.9,
        'wind_speed_lag6': wind_speed * 0.8,
        'wind_speed_lag7': wind_speed * 0.7,
        'wind_speed_lag14': wind_speed * 0.5,

        # Cumulative features - tp
        'tp_cumsum_1d': tp * 8,  # 8 * 3-hourly
        'tp_cumsum_3d': tp * 24,
        'tp_cumsum_7d': tp * 56,
        'tp_cumsum_14d': tp * 112,
        'tp_cumsum_30d': tp * 240,

        # Change features - tp
        'tp_change_1d': tp * 0.1,
        'tp_change_3d': tp * 0.2,
        'tp_intensity_3d': tp if tp > 0 else 0,
        'tp_intensity_7d': tp * 0.8 if tp > 0 else 0,

        # Other derived features
        'ro_change_1d': ro * 0.1,
        'ro_cumsum_3d': ro * 24,
        'swvl1_change_1d': 0.0,
        'swvl1_change_3d': swvl1 * 0.02,
        'swvl1_saturation_7d': min(swvl1 * 1.1, 0.5),
        'soil_rainfall_interaction': swvl1 * tp * 100,

        # Temperature features
        't2m_anomaly_7d': 0.0,
        't2m_anomaly_30d': 0.0,
        't2m_change_1d': 0.5,
        't2m_change_3d': 1.0,

        # Wind features
        'wind_speed_change_1d': wind_speed * 0.1,
        'wind_speed_change_3d': wind_speed * 0.2,
        'wind_speed_anomaly_7d': 0.0,
        'wind_accel_1d': 0.0,

        # Temporal features
        'day_of_year': basic_input['day_of_year'],
        'month': basic_input['month'],
        'day_of_week': basic_input['day_of_week'],
        'week_of_year': basic_input['week_of_year'],
        'month_sin': basic_input['month_sin'],
        'month_cos': basic_input['month_cos'],
        'day_of_year_sin': basic_input['day_of_year_sin'],
        'day_of_year_cos': basic_input['day_of_year_cos'],
        'day_of_week_sin': basic_input['day_of_week_sin'],
        'day_of_week_cos': basic_input['day_of_week_cos'],
    }

    return features

# Predict function for manual input
def make_prediction(basic_input, models, preprocessed_folds, fold=3, threshold=0.5):
    """Make prediction using specified fold models"""
    # Generate all 81 features
    all_features = generate_features_from_basic_input(basic_input)

    # Convert to DataFrame with correct column order matching scaler
    feature_order = [
        'tp', 't2m', 'u10', 'v10', 'swvl1', 'wind_speed',
        'tp_lag1', 'tp_lag2', 'tp_lag3', 'tp_lag6', 'tp_lag7', 'tp_lag14',
        'ro_lag1', 'ro_lag2', 'ro_lag3', 'ro_lag6', 'ro_lag7', 'ro_lag14',
        't2m_lag1', 't2m_lag2', 't2m_lag3', 't2m_lag6', 't2m_lag7', 't2m_lag14',
        'u10_lag1', 'u10_lag2', 'u10_lag3', 'u10_lag6', 'u10_lag7', 'u10_lag14',
        'v10_lag1', 'v10_lag2', 'v10_lag3', 'v10_lag6', 'v10_lag7', 'v10_lag14',
        'swvl1_lag1', 'swvl1_lag2', 'swvl1_lag3', 'swvl1_lag6', 'swvl1_lag7', 'swvl1_lag14',
        'wind_speed_lag1', 'wind_speed_lag2', 'wind_speed_lag3', 'wind_speed_lag6', 'wind_speed_lag7', 'wind_speed_lag14',
        'tp_cumsum_1d', 'tp_cumsum_3d', 'tp_cumsum_7d', 'tp_cumsum_14d', 'tp_cumsum_30d',
        'tp_change_1d', 'tp_change_3d', 'tp_intensity_3d', 'tp_intensity_7d',
        'ro_change_1d', 'ro_cumsum_3d',
        'swvl1_change_1d', 'swvl1_change_3d', 'swvl1_saturation_7d', 'soil_rainfall_interaction',
        't2m_anomaly_7d', 't2m_anomaly_30d', 't2m_change_1d', 't2m_change_3d',
        'wind_speed_change_1d', 'wind_speed_change_3d', 'wind_speed_anomaly_7d', 'wind_accel_1d',
        'day_of_year', 'month', 'day_of_week', 'week_of_year',
        'month_sin', 'month_cos', 'day_of_year_sin', 'day_of_year_cos', 'day_of_week_sin', 'day_of_week_cos'
    ]

    input_df = pd.DataFrame([all_features])[feature_order]

    # Apply scaler from training
    scaler = preprocessed_folds[fold - 1]['scaler']
    input_scaled = scaler.transform(input_df)

    # Get models for specified fold
    xgb_model = models[f'xgb_fold{fold}']
    lstm_model = models[f'lstm_fold{fold}']

    # Make predictions
    xgb_proba = xgb_model.predict_proba(input_scaled)[:, 1][0]
    lstm_proba = lstm_model.predict_proba(input_scaled)[:, 1][0]

    # Ensemble (soft voting: average probabilities)
    ensemble_proba = 0.45 * xgb_proba + 0.55 * lstm_proba

    return {
        'xgb_proba': xgb_proba,
        'lstm_proba': lstm_proba,
        'ensemble_proba': ensemble_proba,
        'prediction': 1 if ensemble_proba >= threshold else 0
    }

# CSS styling
st.markdown("""
<style>
    .metric-box {
        background-color: #f5f5f4;
        padding: 18px 20px;
        border-radius: 4px;
        margin: 10px 0;
        text-align: left;
        border: 1px solid #e2e2df;
    }
    .status-card {
        padding: 18px 22px;
        border-radius: 4px;
        margin: 14px 0;
        text-align: left;
        border-left: 3px solid;
        background-color: #fafaf9;
    }
    .status-card.alert {
        border-left-color: #b3441e;
    }
    .status-card.normal {
        border-left-color: #3f6b4a;
    }
    .status-card .status-label {
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #6b6b68;
        margin-bottom: 4px;
    }
    .status-card .status-value {
        font-size: 22px;
        font-weight: 600;
        color: #1f1f1d;
    }
    .status-card .status-detail {
        font-size: 14px;
        color: #4a4a47;
        margin-top: 4px;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title('Sistem Prediksi Hujan Ekstrem Aceh')
st.caption('XGBoost + LSTM Ensemble — Data Reanalisis ERA5')

# Load data
try:
    models = load_all_models()
    model_results, preprocessed_folds, interpretation = load_model_results()
    predictions = load_predictions()
    best_fold = model_results['results_all_folds'][-1]
    optimal_threshold = interpretation['optimal_threshold']
except Exception as e:
    st.error(f'Error loading data: {str(e)}')
    st.stop()

# Sidebar navigation
st.sidebar.markdown('### Navigasi')
page = st.sidebar.radio(
    'Pilih halaman',
    ['Dashboard', 'Prediksi Manual', 'Analisis Model', 'Feature Importance', 'Detail Prediksi'],
    label_visibility='collapsed'
)

# ============================================================================
# PAGE 1: DASHBOARD
# ============================================================================
if page == 'Dashboard':
    st.markdown('---')

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            'Akurasi',
            f'{best_fold["ensemble_metrics"]["accuracy"]:.2%}',
            'Ensemble Fold 3'
        )

    with col2:
        st.metric(
            'Precision',
            f'{best_fold["ensemble_metrics"]["precision"]:.2%}',
            'Class Ekstrem'
        )

    with col3:
        st.metric(
            'Recall',
            f'{best_fold["ensemble_metrics"]["recall"]:.2%}',
            'Deteksi Event'
        )

    with col4:
        st.metric(
            'F1-Score',
            f'{best_fold["ensemble_metrics"]["f1"]:.4f}',
            'Balanced Metric'
        )

    st.markdown('---')

    # Performance comparison
    st.subheader('Performa Model (3-Fold Cross-Validation)')

    # Calculate stats
    xgb_stats = calc_metrics_stats(model_results['xgb_scores'])
    lstm_stats = calc_metrics_stats(model_results['lstm_scores'])
    ensemble_stats = calc_metrics_stats(model_results['ensemble_scores'])

    metrics_data = {
        'Model': ['XGBoost', 'LSTM', 'Ensemble'],
        'Accuracy': [
            f"{xgb_stats['accuracy_mean']:.4f} ± {xgb_stats['accuracy_std']:.4f}",
            f"{lstm_stats['accuracy_mean']:.4f} ± {lstm_stats['accuracy_std']:.4f}",
            f"{ensemble_stats['accuracy_mean']:.4f} ± {ensemble_stats['accuracy_std']:.4f}"
        ],
        'Precision': [
            f"{xgb_stats['precision_mean']:.4f} ± {xgb_stats['precision_std']:.4f}",
            f"{lstm_stats['precision_mean']:.4f} ± {lstm_stats['precision_std']:.4f}",
            f"{ensemble_stats['precision_mean']:.4f} ± {ensemble_stats['precision_std']:.4f}"
        ],
        'Recall': [
            f"{xgb_stats['recall_mean']:.4f} ± {xgb_stats['recall_std']:.4f}",
            f"{lstm_stats['recall_mean']:.4f} ± {lstm_stats['recall_std']:.4f}",
            f"{ensemble_stats['recall_mean']:.4f} ± {ensemble_stats['recall_std']:.4f}"
        ],
        'F1-Score': [
            f"{xgb_stats['f1_mean']:.4f} ± {xgb_stats['f1_std']:.4f}",
            f"{lstm_stats['f1_mean']:.4f} ± {lstm_stats['f1_std']:.4f}",
            f"{ensemble_stats['f1_mean']:.4f} ± {ensemble_stats['f1_std']:.4f}"
        ],
        'AUC-ROC': [
            f"{xgb_stats['auc_mean']:.4f} ± {xgb_stats['auc_std']:.4f}",
            f"{lstm_stats['auc_mean']:.4f} ± {lstm_stats['auc_std']:.4f}",
            f"{ensemble_stats['auc_mean']:.4f} ± {ensemble_stats['auc_std']:.4f}"
        ]
    }

    st.dataframe(pd.DataFrame(metrics_data), use_container_width=True)

    st.markdown('---')

    # Visualizations
    col1, col2 = st.columns(2)

    with col1:
        st.markdown('**Confusion Matrix (Fold 3)**')
        from sklearn.metrics import confusion_matrix

        cm = confusion_matrix(predictions['Actual'], predictions['Ensemble'])

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                   xticklabels=['Non-Ekstrem', 'Ekstrem'],
                   yticklabels=['Non-Ekstrem', 'Ekstrem'],
                   ax=ax, annot_kws={'size': 15, 'weight': 'medium'},
                   linewidths=0.5, linecolor='#e2e2df')
        ax.set_ylabel('Actual', fontsize=11)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_title('Confusion Matrix — Ensemble', fontsize=12, fontweight='medium')
        st.pyplot(fig)

    with col2:
        st.markdown('**ROC Curve (Fold 3)**')
        from sklearn.metrics import roc_curve, auc

        fpr, tpr, _ = roc_curve(predictions['Actual'], predictions['Ensemble_Proba'])
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color='#1f4e5f', lw=2,
               label=f'Ensemble (AUC = {roc_auc:.3f})')
        ax.plot([0, 1], [0, 1], color='#9a9a96', lw=1.2, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title('ROC Curve', fontsize=12, fontweight='medium')
        ax.legend(loc="lower right", frameon=False)
        ax.grid(True, alpha=0.2)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        st.pyplot(fig)

# ============================================================================
# PAGE 2: PREDIKSI MANUAL
# ============================================================================
elif page == 'Prediksi Manual':
    st.markdown('---')
    st.subheader('Input Data untuk Prediksi')

    st.info('Masukkan data meteorologi dasar. Sistem otomatis menghasilkan fitur tambahan (lag, cumulative, dsb.) dengan asumsi kondisi stabil.')

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('**Kondisi Meteorologi Saat Ini**')
        tp = st.number_input('Total Precipitation (tp) [m]', min_value=0.0, max_value=0.1, value=0.001, step=0.0001, format="%.6f", help="Akumulasi hujan 3 jam terakhir")
        ro = st.number_input('Runoff (ro) [m]', min_value=0.0, max_value=0.01, value=0.0001, step=0.00001, format="%.6f", help="Runoff permukaan")
        t2m = st.number_input('Temperature 2m (t2m) [K]', min_value=280.0, max_value=310.0, value=298.0, step=0.5, help="Suhu 2 meter dari permukaan")
        swvl1 = st.number_input('Soil Moisture (swvl1) [m³/m³]', min_value=0.0, max_value=0.5, value=0.35, step=0.01, help="Kelembapan tanah layer 1")

        st.markdown('**Kondisi Angin**')
        u10 = st.number_input('U-wind component (u10) [m/s]', min_value=-20.0, max_value=20.0, value=2.0, step=0.5, help="Komponen angin timur-barat")
        v10 = st.number_input('V-wind component (v10) [m/s]', min_value=-20.0, max_value=20.0, value=1.5, step=0.5, help="Komponen angin utara-selatan")
        wind_speed = np.sqrt(u10**2 + v10**2)
        st.metric('Wind Speed Total', f'{wind_speed:.2f} m/s')

    with col2:
        st.markdown('**Informasi Waktu**')
        now = datetime.now()

        selected_date = st.date_input('Tanggal', value=now.date())
        selected_hour = st.slider('Jam', 0, 23, now.hour)

        # Calculate temporal features
        day_of_year = selected_date.timetuple().tm_yday
        month = selected_date.month
        day_of_week = selected_date.weekday()
        week_of_year = selected_date.isocalendar()[1]

        # Cyclic encoding
        month_sin = np.sin(2 * np.pi * month / 12)
        month_cos = np.cos(2 * np.pi * month / 12)
        day_of_year_sin = np.sin(2 * np.pi * day_of_year / 365)
        day_of_year_cos = np.cos(2 * np.pi * day_of_year / 365)
        day_of_week_sin = np.sin(2 * np.pi * day_of_week / 7)
        day_of_week_cos = np.cos(2 * np.pi * day_of_week / 7)

        st.info(f'''
        **Informasi Temporal**
        - Hari ke-{day_of_year} dalam tahun
        - Bulan: {month} ({selected_date.strftime("%B")})
        - Hari: {selected_date.strftime("%A")}
        - Minggu ke-{week_of_year}
        ''')

        st.markdown('**Catatan**')
        st.warning('''
        **Asumsi untuk prediksi:**
        - Lag features: diasumsikan kondisi serupa di masa lalu dengan penurunan bertahap
        - Cumulative features: dihitung berdasarkan kondisi saat ini
        - Change features: diasumsikan perubahan moderat

        Untuk prediksi akurat, gunakan data observasi historis lengkap.
        ''')

    # Prepare basic input
    basic_input = {
        'tp': tp,
        'ro': ro,
        't2m': t2m,
        'u10': u10,
        'v10': v10,
        'swvl1': swvl1,
        'wind_speed': wind_speed,
        'day_of_year': day_of_year,
        'month': month,
        'day_of_week': day_of_week,
        'week_of_year': week_of_year,
        'month_sin': month_sin,
        'month_cos': month_cos,
        'day_of_year_sin': day_of_year_sin,
        'day_of_year_cos': day_of_year_cos,
        'day_of_week_sin': day_of_week_sin,
        'day_of_week_cos': day_of_week_cos,
    }

    st.markdown('---')

    # Prediction button
    if st.button('Prediksi Sekarang', type='primary', use_container_width=True):
        with st.spinner('Generating features dan melakukan prediksi...'):
            try:
                result = make_prediction(basic_input, models, preprocessed_folds, fold=3, threshold=optimal_threshold)

                st.markdown('---')
                st.subheader('Hasil Prediksi')

                # Display probabilities
                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric('XGBoost Probability', f'{result["xgb_proba"]:.4f}')
                with col2:
                    st.metric('LSTM Probability', f'{result["lstm_proba"]:.4f}')
                with col3:
                    st.metric('Ensemble Probability', f'{result["ensemble_proba"]:.4f}')

                # Final prediction
                st.markdown('---')
                threshold = optimal_threshold
                is_extreme = result['ensemble_proba'] >= threshold

                if is_extreme:
                    st.markdown(f'''
                    <div class="status-card alert">
                        <div class="status-label">Status</div>
                        <div class="status-value">Hujan Ekstrem Diprediksi — Waspada</div>
                        <div class="status-detail">Probabilitas: {result["ensemble_proba"]:.1%}</div>
                    </div>
                    ''', unsafe_allow_html=True)

                    st.warning('''
                    **Rekomendasi:**
                    - Tetap waspada terhadap potensi banjir bandang
                    - Hindari aktivitas outdoor di area rawan
                    - Monitor update cuaca setiap jam
                    - Siapkan jalur evakuasi dan perlengkapan darurat
                    - Informasikan ke warga sekitar
                    ''')
                else:
                    st.markdown(f'''
                    <div class="status-card normal">
                        <div class="status-label">Status</div>
                        <div class="status-value">Kondisi Aman — Normal</div>
                        <div class="status-detail">Probabilitas Ekstrem: {result["ensemble_proba"]:.1%}</div>
                    </div>
                    ''', unsafe_allow_html=True)

                    st.success('''
                    **Status:**
                    - Tidak ada indikasi hujan ekstrem dalam 3 jam ke depan
                    - Aktivitas normal dapat dilanjutkan
                    - Tetap monitor kondisi cuaca secara berkala
                    - Waspadai perubahan cuaca mendadak
                    ''')

                # Probability gauge
                st.markdown('---')
                st.markdown('**Indikator Tingkat Risiko**')

                fig, ax = plt.subplots(figsize=(12, 1.8))

                # Draw gauge as a single-hue gradient (calm, not a traffic-light rainbow)
                band_colors = ['#dce8e2', '#b9cfc2', '#8fae9c', '#5f8a74']
                labels = ['Aman', 'Hati-hati', 'Waspada', 'Bahaya']
                bounds = [0, 0.25, 0.50, 0.75, 1.0]

                for i in range(len(band_colors)):
                    ax.barh(0, bounds[i+1] - bounds[i], left=bounds[i],
                           height=0.5, color=band_colors[i], edgecolor='white', linewidth=1.5)
                    mid = (bounds[i] + bounds[i+1]) / 2
                    ax.text(mid, 0, labels[i], ha='center', va='center',
                           fontsize=10, color='#2a2a28')

                # Mark ensemble probability
                ax.plot([result['ensemble_proba'], result['ensemble_proba']], [-0.4, 0.9],
                       color='#1f4e5f', linewidth=3, label=f'Ensemble: {result["ensemble_proba"]:.3f}', zorder=10)
                ax.scatter([result['ensemble_proba']], [0], s=140, c='#1f4e5f', marker='v',
                          edgecolors='white', linewidths=1.5, zorder=11)

                # Mark threshold
                ax.axvline(x=threshold, color='#b3441e', linestyle='--', linewidth=1.5,
                          label=f'Threshold: {threshold:.3f}', alpha=0.9)

                ax.set_xlim([0, 1])
                ax.set_ylim([-0.6, 1.2])
                ax.set_xlabel('Probabilitas Hujan Ekstrem', fontsize=11)
                ax.set_yticks([])
                ax.legend(loc='upper left', fontsize=9, frameon=False)
                for spine in ['top', 'right', 'left']:
                    ax.spines[spine].set_visible(False)

                plt.tight_layout()
                st.pyplot(fig)

            except Exception as e:
                st.error(f'Error during prediction: {str(e)}')
                import traceback
                st.code(traceback.format_exc())

# ============================================================================
# PAGE 3: ANALISIS MODEL
# ============================================================================
elif page == 'Analisis Model':
    st.markdown('---')
    st.subheader('Analisis Performa Model')

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('**Confusion Matrix (Fold 3)**')
        from sklearn.metrics import confusion_matrix, classification_report

        cm = confusion_matrix(predictions['Actual'], predictions['Ensemble'])

        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                   xticklabels=['Non-Ekstrem (0)', 'Ekstrem (1)'],
                   yticklabels=['Non-Ekstrem (0)', 'Ekstrem (1)'],
                   ax=ax, annot_kws={'size': 15, 'weight': 'medium'},
                   linewidths=0.5, linecolor='#e2e2df')
        ax.set_ylabel('Actual', fontsize=11)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_title('Confusion Matrix', fontsize=12, fontweight='medium')
        st.pyplot(fig)

        # Classification report
        st.markdown('#### Classification Report')
        report = classification_report(predictions['Actual'], predictions['Ensemble'],
                                       target_names=['Non-Ekstrem', 'Ekstrem'],
                                       output_dict=True)
        st.json(report)

    with col2:
        st.markdown('#### Probability Distribution')

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.hist(predictions[predictions['Actual'] == 0]['Ensemble_Proba'],
               bins=50, alpha=0.7, label='Actual Non-Ekstrem', color='#7a9e8c')
        ax.hist(predictions[predictions['Actual'] == 1]['Ensemble_Proba'],
               bins=50, alpha=0.7, label='Actual Ekstrem', color='#b3441e')
        ax.axvline(x=optimal_threshold, color='#1f4e5f', linestyle='--', linewidth=1.5,
                  label=f'Optimal Threshold ({optimal_threshold:.3f})')
        ax.set_xlabel('Probability', fontsize=11)
        ax.set_ylabel('Frequency', fontsize=11)
        ax.set_title('Prediction Probability Distribution', fontsize=12, fontweight='medium')
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.2)
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        st.pyplot(fig)

        st.markdown('**Model Comparison**')

        # Compare all models
        models_comparison = {
            'Model': ['XGBoost', 'LSTM', 'Ensemble'],
            'True Positives': [
                (predictions['XGBoost'] & predictions['Actual']).sum(),
                (predictions['LSTM'] & predictions['Actual']).sum(),
                (predictions['Ensemble'] & predictions['Actual']).sum()
            ],
            'False Positives': [
                (predictions['XGBoost'] & ~predictions['Actual'].astype(bool)).sum(),
                (predictions['LSTM'] & ~predictions['Actual'].astype(bool)).sum(),
                (predictions['Ensemble'] & ~predictions['Actual'].astype(bool)).sum()
            ],
            'False Negatives': [
                (~predictions['XGBoost'].astype(bool) & predictions['Actual'].astype(bool)).sum(),
                (~predictions['LSTM'].astype(bool) & predictions['Actual'].astype(bool)).sum(),
                (~predictions['Ensemble'].astype(bool) & predictions['Actual'].astype(bool)).sum()
            ]
        }

        st.dataframe(pd.DataFrame(models_comparison), use_container_width=True)

# ============================================================================
# PAGE 4: FEATURE IMPORTANCE
# ============================================================================
elif page == 'Feature Importance':
    st.markdown('---')
    st.subheader('Analisis Feature Importance')

    feature_importance = interpretation['feature_importance']

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('**Top Important Features**')

        top_n = st.slider('Jumlah fitur', 5, 16, 10)

        fig, ax = plt.subplots(figsize=(8, 6))
        top_features = feature_importance.head(top_n)
        ax.barh(range(len(top_features)), top_features['Importance'].values,
               color='#1f4e5f', alpha=0.85)
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['Feature'].values)
        ax.set_xlabel('Importance Score', fontsize=11)
        ax.set_title(f'Top {top_n} Features (XGBoost)', fontsize=12, fontweight='medium')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.2, axis='x')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

    with col2:
        st.markdown('**Feature Correlations**')

        correlations = interpretation['feature_correlations']

        fig, ax = plt.subplots(figsize=(8, 6))
        top_corr = pd.concat([correlations.head(8), correlations.tail(8)])
        colors = ['#5f8a74' if x > 0 else '#b3441e' for x in top_corr.values]
        ax.barh(range(len(top_corr)), top_corr.values, color=colors, alpha=0.85)
        ax.set_yticks(range(len(top_corr)))
        ax.set_yticklabels(top_corr.index)
        ax.set_xlabel('Correlation with Target', fontsize=11)
        ax.set_title('Feature-Target Correlation', fontsize=12, fontweight='medium')
        ax.axvline(x=0, color='#4a4a47', linestyle='-', linewidth=0.8)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.2, axis='x')
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown('---')
    st.markdown('**Feature Details Table**')
    st.dataframe(feature_importance, use_container_width=True)

# ============================================================================
# PAGE 5: DETAIL PREDIKSI
# ============================================================================
elif page == 'Detail Prediksi':
    st.markdown('---')
    st.subheader('Detail Prediksi (Fold 3 - Test Set)')

    # Filters
    col1, col2, col3 = st.columns(3)

    with col1:
        filter_type = st.selectbox(
            'Filter',
            ['Semua', 'Ekstrem (Actual)', 'Non-Ekstrem (Actual)',
             'False Positives', 'False Negatives', 'True Positives']
        )

    with col2:
        min_proba = st.slider('Min Probability', 0.0, 1.0, 0.0)

    with col3:
        max_rows = st.slider('Rows', 10, 100, 20)

    # Apply filters
    filtered = predictions.copy()

    if filter_type == 'Ekstrem (Actual)':
        filtered = filtered[filtered['Actual'] == 1]
    elif filter_type == 'Non-Ekstrem (Actual)':
        filtered = filtered[filtered['Actual'] == 0]
    elif filter_type == 'False Positives':
        filtered = filtered[(filtered['Ensemble'] == 1) & (filtered['Actual'] == 0)]
    elif filter_type == 'False Negatives':
        filtered = filtered[(filtered['Ensemble'] == 0) & (filtered['Actual'] == 1)]
    elif filter_type == 'True Positives':
        filtered = filtered[(filtered['Ensemble'] == 1) & (filtered['Actual'] == 1)]

    filtered = filtered[filtered['Ensemble_Proba'] >= min_proba]

    st.dataframe(filtered.head(max_rows), use_container_width=True)
    st.markdown(f'**Total: {len(filtered)} records**')

    # Download
    st.markdown('---')
    csv = predictions.to_csv(index=False)
    st.download_button(
        'Download All Predictions',
        data=csv,
        file_name=f'predictions_{datetime.now().strftime("%Y%m%d_%H%M")}.csv',
        mime='text/csv'
    )

# Footer
st.markdown('---')
st.markdown(f'''
<div style="text-align: center; color: #666;">
<b>Sistem Prediksi Hujan Ekstrem Aceh</b><br>
ERA5 Reanalysis 2020-2024 | 3-Fold CV | XGBoost + LSTM Ensemble<br>
&copy; 2024 | Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
</div>
''', unsafe_allow_html=True)
