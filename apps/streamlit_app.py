import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configure page
st.set_page_config(
    page_title='Prediksi Banjir Aceh',
    page_icon='🌊',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Load models and data
@st.cache_resource
def load_models():
    model_results_path = 'd:/S2/prediksi - hujan/trained_models_results.pkl'
    with open(model_results_path, 'rb') as f:
        model_results = pickle.load(f)
    
    pkl_path = 'd:/S2/prediksi - hujan/preprocessed_data_folds.pkl'
    with open(pkl_path, 'rb') as f:
        preprocessed_folds = pickle.load(f)
    
    interp_path = 'd:/S2/prediksi - hujan/model_interpretation_results.pkl'
    with open(interp_path, 'rb') as f:
        interpretation = pickle.load(f)
    
    return model_results, preprocessed_folds, interpretation

# Load interpretation for threshold
@st.cache_resource
def load_interpretation():
    interp_path = 'd:/S2/prediksi - hujan/model_interpretation_results.pkl'
    with open(interp_path, 'rb') as f:
        interpretation = pickle.load(f)
    return interpretation

# Load prediction data
@st.cache_data
def load_predictions():
    pred_path = 'd:/S2/prediksi - hujan/fold3_predictions.csv'
    predictions = pd.read_csv(pred_path)
    return predictions

# CSS styling
st.markdown("""
<style>
    .metric-box {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        text-align: center;
    }
    .metric-value {
        font-size: 28px;
        font-weight: bold;
        color: #1f77b4;
    }
    .metric-label {
        font-size: 14px;
        color: #666;
        margin-top: 5px;
    }
    .warning-box {
        background-color: #ff4444;
        color: white;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .safe-box {
        background-color: #44aa44;
        color: white;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title('🌊 Sistem Prediksi Banjir Aceh')
st.markdown('**Powered by XGBoost + LSTM Ensemble | ERA5 Data**')

# Load data
try:
    model_results, preprocessed_folds, interpretation = load_models()
    predictions = load_predictions()
    best_fold = model_results['results_all_folds'][-1]
    optimal_threshold = interpretation['optimal_threshold']
except Exception as e:
    st.error(f'Error loading models: {str(e)}')
    st.stop()

# Sidebar navigation
st.sidebar.markdown('## 📊 Navigasi')
page = st.sidebar.radio(
    'Pilih halaman:',
    ['🏠 Dashboard Utama', '📈 Analisis Model', '🎯 Interpretasi Fitur', '📋 Prediksi Detail']
)

# ============================================================================
# PAGE 1: DASHBOARD UTAMA
# ============================================================================
if page == '🏠 Dashboard Utama':
    st.markdown('---')
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(
            '🎯 Threshold Optimal',
            f'{optimal_threshold:.4f}',
            'Maksimalisasi F1-Score'
        )
    
    with col2:
        st.metric(
            '✅ Akurasi Model',
            f'{best_fold["ensemble_metrics"]["accuracy"]:.2%}',
            'Ensemble (Best)'
        )
    
    with col3:
        st.metric(
            '📊 F1-Score',
            f'{best_fold["ensemble_metrics"]["f1"]:.4f}',
            'Test 2024'
        )
    
    st.markdown('---')
    
    # Real-time status
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader('📊 Statistik Prediksi (Fold 3 - 366 sample)')
        
        actual_positive = (predictions['Actual'] == 1).sum()
        ensemble_positive = (predictions['Ensemble'] == 1).sum()
        
        stats_data = {
            'Kategori': ['Actual BAHAYA', 'Predicted BAHAYA (Ensemble)', 
                        'XGBoost BAHAYA', 'LSTM BAHAYA'],
            'Jumlah': [
                actual_positive,
                ensemble_positive,
                (predictions['XGBoost'] == 1).sum(),
                (predictions['LSTM'] == 1).sum()
            ]
        }
        stats_df = pd.DataFrame(stats_data)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        bars = ax.bar(stats_df['Kategori'], stats_df['Jumlah'], 
                      color=['#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4'],
                      alpha=0.8)
        ax.set_ylabel('Jumlah Prediksi', fontsize=11)
        ax.set_title('Distribusi Prediksi BAHAYA', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, val) in enumerate(zip(bars, stats_df['Jumlah'])):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{val}', ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        plt.xticks(rotation=15, ha='right')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.subheader('📈 Model Ensemble Composition')
        
        fig, ax = plt.subplots(figsize=(6, 6))
        weights = [45, 55]
        labels = ['XGBoost (45%)', 'LSTM (55%)']
        colors = ['#FF9999', '#66B2FF']
        
        wedges, texts, autotexts = ax.pie(
            weights, labels=labels, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 11, 'weight': 'bold'}
        )
        ax.set_title('Ensemble Weights', fontsize=12, fontweight='bold')
        plt.tight_layout()
        st.pyplot(fig)
    
    st.markdown('---')
    
    # Performance metrics
    st.subheader('📊 Metrik Performa (3-Fold Cross-Validation)')
    
    xgb_scores = model_results['xgb_scores']
    lstm_scores = model_results['lstm_scores']
    ensemble_scores = model_results['ensemble_scores']
    
    metrics_data = {
        'Model': ['XGBoost', 'LSTM', 'Ensemble'],
        'Accuracy': [
            f"{xgb_scores['accuracy_mean']:.4f} ± {xgb_scores['accuracy_std']:.4f}",
            f"{lstm_scores['accuracy_mean']:.4f} ± {lstm_scores['accuracy_std']:.4f}",
            f"{ensemble_scores['accuracy_mean']:.4f} ± {ensemble_scores['accuracy_std']:.4f}"
        ],
        'F1-Score': [
            f"{xgb_scores['f1_mean']:.4f} ± {xgb_scores['f1_std']:.4f}",
            f"{lstm_scores['f1_mean']:.4f} ± {lstm_scores['f1_std']:.4f}",
            f"{ensemble_scores['f1_mean']:.4f} ± {ensemble_scores['f1_std']:.4f}"
        ],
        'AUC-ROC': [
            f"{xgb_scores['auc_mean']:.4f} ± {xgb_scores['auc_std']:.4f}",
            f"{lstm_scores['auc_mean']:.4f} ± {lstm_scores['auc_std']:.4f}",
            f"{ensemble_scores['auc_mean']:.4f} ± {ensemble_scores['auc_std']:.4f}"
        ]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    st.dataframe(metrics_df, use_container_width=True)

# ============================================================================
# PAGE 2: ANALISIS MODEL
# ============================================================================
elif page == '📈 Analisis Model':
    st.markdown('---')
    st.subheader('🔍 Analisis Performa Model')
    
    col1, col2 = st.columns(2)
    
    # Confusion Matrix untuk Ensemble
    with col1:
        st.markdown('#### Ensemble - Confusion Matrix')
        from sklearn.metrics import confusion_matrix
        
        ensemble_pred = predictions['Ensemble'].values
        cm = confusion_matrix(predictions['Actual'], ensemble_pred)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                   xticklabels=['AMAN', 'BAHAYA'],
                   yticklabels=['AMAN', 'BAHAYA'],
                   ax=ax, annot_kws={'size': 14, 'weight': 'bold'})
        ax.set_ylabel('Actual', fontsize=11)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_title('Confusion Matrix (Ensemble)', fontsize=12, fontweight='bold')
        st.pyplot(fig)
        
        # Metrics
        from sklearn.metrics import precision_recall_fscore_support
        precision, recall, f1, _ = precision_recall_fscore_support(
            predictions['Actual'], ensemble_pred, average='weighted'
        )
        
        st.markdown(f'''
        **Metrik Ensemble:**
        - Precision: {precision:.4f}
        - Recall: {recall:.4f}
        - F1-Score: {f1:.4f}
        ''')
    
    # ROC Curve
    with col2:
        st.markdown('#### ROC Curve')
        from sklearn.metrics import roc_curve, auc
        
        fpr, tpr, _ = roc_curve(predictions['Actual'], predictions['Ensemble_Proba'])
        roc_auc = auc(fpr, tpr)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, color='darkorange', lw=2, 
               label=f'Ensemble (AUC = {roc_auc:.4f})')
        ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel('False Positive Rate', fontsize=11)
        ax.set_ylabel('True Positive Rate', fontsize=11)
        ax.set_title('ROC Curve', fontsize=12, fontweight='bold')
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
    
    st.markdown('---')
    
    # Probability distribution
    st.markdown('#### 📊 Distribusi Probabilitas Prediksi')
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(predictions[predictions['Actual'] == 0]['Ensemble_Proba'], 
           bins=50, alpha=0.6, label='Actual AMAN', color='green')
    ax.hist(predictions[predictions['Actual'] == 1]['Ensemble_Proba'], 
           bins=50, alpha=0.6, label='Actual BAHAYA', color='red')
    ax.axvline(x=optimal_threshold, color='blue', linestyle='--', linewidth=2,
              label=f'Optimal Threshold ({optimal_threshold:.4f})')
    ax.set_xlabel('Probability', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Ensemble Prediction Probability Distribution', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    st.pyplot(fig)

# ============================================================================
# PAGE 3: INTERPRETASI FITUR
# ============================================================================
elif page == '🎯 Interpretasi Fitur':
    st.markdown('---')
    st.subheader('🔍 Feature Importance & Analysis')
    
    col1, col2 = st.columns(2)
    
    # Feature importance
    with col1:
        st.markdown('#### Top Features (XGBoost)')
        
        feature_importance = interpretation['feature_importance']
        top_n = st.slider('Jumlah fitur teratas', 5, 20, 15)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        top_features = feature_importance.head(top_n)
        ax.barh(range(len(top_features)), top_features['Importance'].values, 
               alpha=0.8, color='steelblue')
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['Feature'].values, fontsize=10)
        ax.set_xlabel('Importance', fontsize=11)
        ax.set_title(f'Top {top_n} Important Features', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    # Feature correlations
    with col2:
        st.markdown('#### Korelasi dengan Target')
        
        correlations = interpretation['feature_correlations']
        top_pos = st.slider('Top Positive Correlations', 3, 10, 5, key='pos_corr')
        
        fig, ax = plt.subplots(figsize=(8, 6))
        top_corr = pd.concat([
            correlations.head(top_pos),
            correlations.tail(top_pos)
        ])
        colors = ['green' if x > 0 else 'red' for x in top_corr.values]
        ax.barh(range(len(top_corr)), top_corr.values, color=colors, alpha=0.7)
        ax.set_yticks(range(len(top_corr)))
        ax.set_yticklabels(top_corr.index, fontsize=10)
        ax.set_xlabel('Correlation', fontsize=11)
        ax.set_title('Feature-Target Correlation', fontsize=12, fontweight='bold')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        st.pyplot(fig)
    
    st.markdown('---')
    
    # Feature details table
    st.markdown('#### 📊 Feature Details')
    
    feature_details = feature_importance.copy()
    st.dataframe(feature_details.head(20), use_container_width=True)

# ============================================================================
# PAGE 4: PREDIKSI DETAIL
# ============================================================================
elif page == '📋 Prediksi Detail':
    st.markdown('---')
    st.subheader('📊 Detailed Predictions')
    
    # Filter options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        filter_type = st.selectbox(
            'Filter Tipe',
            ['Semua', 'BAHAYA (Actual)', 'AMAN (Actual)', 'False Positives', 'False Negatives']
        )
    
    with col2:
        min_proba = st.slider('Minimum Probability', 0.0, 1.0, 0.0)
    
    with col3:
        max_rows = st.slider('Rows to Display', 5, 100, 20)
    
    # Apply filters
    filtered_pred = predictions.copy()
    
    if filter_type == 'BAHAYA (Actual)':
        filtered_pred = filtered_pred[filtered_pred['Actual'] == 1]
    elif filter_type == 'AMAN (Actual)':
        filtered_pred = filtered_pred[filtered_pred['Actual'] == 0]
    elif filter_type == 'False Positives':
        filtered_pred = filtered_pred[
            (filtered_pred['Ensemble'] == 1) & (filtered_pred['Actual'] == 0)
        ]
    elif filter_type == 'False Negatives':
        filtered_pred = filtered_pred[
            (filtered_pred['Ensemble'] == 0) & (filtered_pred['Actual'] == 1)
        ]
    
    filtered_pred = filtered_pred[filtered_pred['Ensemble_Proba'] >= min_proba]
    
    # Display table
    st.dataframe(filtered_pred.head(max_rows).reset_index(drop=True), 
                use_container_width=True)
    
    st.markdown(f'**Total matched: {len(filtered_pred)} records**')
    
    # Download predictions
    st.markdown('---')
    col1, col2 = st.columns(2)
    
    with col1:
        csv = predictions.to_csv(index=False)
        st.download_button(
            label='📥 Download All Predictions (CSV)',
            data=csv,
            file_name=f'predictions_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv',
            mime='text/csv'
        )

# Footer
st.markdown('---')
st.markdown('''
**Catatan:**
- Model dilatih menggunakan ERA5 data (2020-2024)
- 3-Fold time-series cross-validation
- Threshold dioptimalkan untuk maksimalisasi F1-Score
- Update prediksi setiap hari pukul 00:00 WIB
''')
st.markdown(f'**Last updated:** {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
