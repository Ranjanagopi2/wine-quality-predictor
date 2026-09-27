import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# ---------------------------------------------------------
# Page Configuration & Custom CSS
# ---------------------------------------------------------
st.set_page_config(
    page_title="Wine Quality AI Predictor",
    page_icon="🍷",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Burgundy & Modern Theme)
st.markdown("""
<style>
    /* Global Font & Background adjustments */
    .stApp {
        background-color: #0e1117;
        color: #e0e0e0;
    }
    
    /* Header Container */
    .header-container {
        background: linear-gradient(135deg, #4A0E17 0%, #722F37 50%, #2D080C 100%);
        padding: 2.2rem 2rem;
        border-radius: 16px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px 0 rgba(114, 47, 55, 0.37);
        border: 1px solid rgba(212, 175, 55, 0.2);
    }
    .header-title {
        font-size: 2.6rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.5px;
        color: #ffffff;
    }
    .header-subtitle {
        font-size: 1.15rem;
        color: #F3E5AB;
        margin-top: 0.5rem;
        font-weight: 400;
    }

    /* Metric Card Styling */
    .metric-card {
        background: #1e222d;
        border-radius: 12px;
        padding: 1.2rem;
        border-left: 5px solid #722F37;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #F3E5AB;
    }
    .metric-label {
        font-size: 0.9rem;
        color: #9aa0a6;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Prediction Result Cards */
    .prediction-box-high {
        background: linear-gradient(135deg, #1b4332 0%, #2d6a4f 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #52b788;
        color: #d8f3dc;
    }
    .prediction-box-med {
        background: linear-gradient(135deg, #7f5539 0%, #9c6644 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #ddb892;
        color: #ffe8d6;
    }
    .prediction-box-low {
        background: linear-gradient(135deg, #6b1426 0%, #800f2f 100%);
        padding: 1.5rem;
        border-radius: 12px;
        text-align: center;
        border: 1px solid #ff758f;
        color: #ffccd5;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Paths & Artifact Loaders
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'winequality-red.csv')
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'wine_quality_model.joblib')

@st.cache_data
def load_dataset():
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH, sep=';' if ';' in open(DATA_PATH).readline() else ',')
        return df
    else:
        st.error("Dataset not found! Please run `python train_model.py` first.")
        st.stop()

@st.cache_resource
def load_trained_pipeline():
    if os.path.exists(MODEL_PATH):
        return joblib.load(MODEL_PATH)
    else:
        st.error("Saved model file not found! Please run `python train_model.py` to generate the model.")
        st.stop()

df = load_dataset()
pipeline_data = load_trained_pipeline()

model = pipeline_data['model']
scaler = pipeline_data['scaler']
feature_names = pipeline_data['feature_names']
saved_metrics = pipeline_data['metrics']
saved_importances = pipeline_data['feature_importances']

# ---------------------------------------------------------
# App Header
# ---------------------------------------------------------
st.markdown("""
<div class="header-container">
    <div class="header-title">🍷 Wine Quality AI Predictor</div>
    <div class="header-subtitle">Machine Learning Powered Analysis of Physicochemical Wine Properties</div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Sidebar Controls & Information
# ---------------------------------------------------------
st.sidebar.image("https://images.unsplash.com/photo-1510812431401-41d2bd2722f3?w=500&auto=format&fit=crop&q=60", use_container_width=True)
st.sidebar.title("🍇 Control Panel")

st.sidebar.markdown(f"**Model Status**: `Ready ({saved_metrics.get('best_algorithm', 'Random Forest')})`")
st.sidebar.markdown(f"**Model Accuracy**: `{saved_metrics.get('accuracy', 0):.2%}`")

st.sidebar.markdown("---")
st.sidebar.subheader("Presets for Quick Testing")
preset = st.sidebar.selectbox(
    "Choose a Wine Profile Preset:",
    ["Custom Input", "Typical Premium Red Wine (Score 7-8)", "Average Table Red Wine (Score 5-6)", "Faulty / High Acidity Red (Score 3-4)"]
)

# Preset Values
preset_dict = {
    "Custom Input": None,
    "Typical Premium Red Wine (Score 7-8)": {
        'fixed acidity': 8.5, 'volatile acidity': 0.28, 'citric acid': 0.49,
        'residual sugar': 2.2, 'chlorides': 0.055, 'free sulfur dioxide': 14.0,
        'total sulfur dioxide': 35.0, 'density': 0.9940, 'pH': 3.25,
        'sulphates': 0.86, 'alcohol': 12.8
    },
    "Average Table Red Wine (Score 5-6)": {
        'fixed acidity': 7.8, 'volatile acidity': 0.58, 'citric acid': 0.20,
        'residual sugar': 2.4, 'chlorides': 0.085, 'free sulfur dioxide': 15.0,
        'total sulfur dioxide': 52.0, 'density': 0.9968, 'pH': 3.32,
        'sulphates': 0.60, 'alcohol': 9.8
    },
    "Faulty / High Acidity Red (Score 3-4)": {
        'fixed acidity': 7.1, 'volatile acidity': 0.98, 'citric acid': 0.04,
        'residual sugar': 4.8, 'chlorides': 0.120, 'free sulfur dioxide': 6.0,
        'total sulfur dioxide': 85.0, 'density': 0.9985, 'pH': 3.55,
        'sulphates': 0.42, 'alcohol': 8.8
    }
}

active_preset = preset_dict[preset]

st.sidebar.markdown("---")
st.sidebar.markdown("💡 **Tip**: Adjust the chemical composition sliders in the predictor tab to see live quality score recalculation.")

# ---------------------------------------------------------
# Tabs Setup
# ---------------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Predict Quality",
    "📊 Exploratory Analysis",
    "🤖 Model Performance & Retraining",
    "📁 Batch Prediction"
])

# =========================================================
# TAB 1: SINGLE PREDICTION
# =========================================================
with tab1:
    st.subheader("🧪 Input Physicochemical Features")
    st.markdown("Specify the laboratory test values for the red wine sample below:")

    col_a, col_b, col_c = st.columns(3)

    def get_val(col, default_val):
        return active_preset[col] if active_preset and col in active_preset else default_val

    with col_a:
        st.markdown("#### 🍋 Acidity & pH")
        fixed_acidity = st.slider(
            "Fixed Acidity (g/dm³ tartaric acid)", 4.0, 16.0, float(get_val('fixed acidity', 8.3)), 0.1,
            help="Tartaric acid concentration. Gives wine its freshness and structure."
        )
        volatile_acidity = st.slider(
            "Volatile Acidity (g/dm³ acetic acid)", 0.10, 1.60, float(get_val('volatile acidity', 0.52)), 0.01,
            help="High levels lead to unpleasant vinegar taste/odour."
        )
        citric_acid = st.slider(
            "Citric Acid (g/dm³)", 0.00, 1.00, float(get_val('citric acid', 0.27)), 0.01,
            help="Adds 'freshness' and flavor complexity."
        )
        pH = st.slider(
            "pH level", 2.70, 4.10, float(get_val('pH', 3.31)), 0.01,
            help="Describes acidity level (0 very acidic, 14 very basic). Most wines are 3.0–3.6."
        )

    with col_b:
        st.markdown("#### 🍬 Sugars & Sulfur")
        residual_sugar = st.slider(
            "Residual Sugar (g/dm³)", 0.90, 15.50, float(get_val('residual sugar', 2.50)), 0.1,
            help="Amount of sugar remaining after fermentation stops."
        )
        free_so2 = st.slider(
            "Free Sulfur Dioxide (mg/dm³)", 1.0, 72.0, float(get_val('free sulfur dioxide', 14.0)), 1.0,
            help="Prevents microbial growth and oxidation of wine."
        )
        total_so2 = st.slider(
            "Total Sulfur Dioxide (mg/dm³)", 6.0, 289.0, float(get_val('total sulfur dioxide', 46.0)), 1.0,
            help="Sum of free and bound SO2. Detectable in nose above 50 ppm."
        )
        chlorides = st.slider(
            "Chlorides (g/dm³ sodium chloride)", 0.010, 0.600, float(get_val('chlorides', 0.087)), 0.001,
            help="Amount of salt in the wine."
        )

    with col_c:
        st.markdown("#### 🍷 Structure & Alcohol")
        density = st.slider(
            "Density (g/cm³)", 0.9900, 1.0037, float(get_val('density', 0.9967)), 0.0001, format="%.4f",
            help="Depends on percent alcohol and sugar content."
        )
        sulphates = st.slider(
            "Sulphates (g/dm³ potassium sulphate)", 0.30, 2.00, float(get_val('sulphates', 0.65)), 0.01,
            help="Wine additive contributing to SO2 levels, acts as antimicrobial/antioxidant."
        )
        alcohol = st.slider(
            "Alcohol (% vol)", 8.0, 15.0, float(get_val('alcohol', 10.4)), 0.1,
            help="Percent alcohol content of the wine. High impact on perceived quality."
        )

    # Compile input frame
    input_dict = {
        'fixed acidity': fixed_acidity,
        'volatile acidity': volatile_acidity,
        'citric acid': citric_acid,
        'residual sugar': residual_sugar,
        'chlorides': chlorides,
        'free sulfur dioxide': free_so2,
        'total sulfur dioxide': total_so2,
        'density': density,
        'pH': pH,
        'sulphates': sulphates,
        'alcohol': alcohol
    }
    input_df = pd.DataFrame([input_dict])[feature_names]

    # Predict
    input_scaled = scaler.transform(input_df)
    predicted_quality = int(model.predict(input_scaled)[0])
    probabilities = model.predict_proba(input_scaled)[0]
    classes = model.classes_

    st.markdown("---")
    st.subheader("🎯 Prediction Results")

    res_col1, res_col2, res_col3 = st.columns([1.2, 1.8, 1.5])

    with res_col1:
        # Category Determination
        if predicted_quality >= 7:
            category = "High Quality (Premium)"
            css_class = "prediction-box-high"
            badge = "🌟"
        elif predicted_quality >= 5:
            category = "Medium Quality (Standard)"
            css_class = "prediction-box-med"
            badge = "🍷"
        else:
            category = "Low Quality (Poor / Faulty)"
            css_class = "prediction-box-low"
            badge = "⚠️"

        st.markdown(f"""
        <div class="{css_class}">
            <h2>{badge} Quality Score: {predicted_quality}/10</h2>
            <h4>Category: {category}</h4>
        </div>
        """, unsafe_allow_html=True)

        st.write("")
        st.markdown("**Predicted Score Gauge:**")
        # Plotly Gauge Chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = predicted_quality,
            domain = {'x': [0, 1], 'y': [0, 1]},
            gauge = {
                'axis': {'range': [3, 9], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#D4AF37"},
                'bgcolor': "#1e222d",
                'borderwidth': 2,
                'bordercolor': "#333",
                'steps': [
                    {'range': [3, 5], 'color': '#6b1426'},
                    {'range': [5, 7], 'color': '#7f5539'},
                    {'range': [7, 9], 'color': '#1b4332'}
                ],
            }
        ))
        fig_gauge.update_layout(
            height=200, margin=dict(l=20, r=20, t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)", font=dict(color="white")
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with res_col2:
        st.markdown("**Probability Distribution per Quality Score:**")
        prob_df = pd.DataFrame({
            'Quality Score': [f"Score {c}" for c in classes],
            'Probability': probabilities
        })

        fig_prob = px.bar(
            prob_df, x='Quality Score', y='Probability',
            text=[f"{p:.1%}" for p in probabilities],
            color='Probability',
            color_continuous_scale=['#4A0E17', '#722F37', '#D4AF37', '#52b788']
        )
        fig_prob.update_layout(
            height=300,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white"),
            showlegend=False,
            margin=dict(l=10, r=10, t=10, b=10)
        )
        fig_prob.update_yaxes(range=[0, 1])
        st.plotly_chart(fig_prob, use_container_width=True)

    with res_col3:
        st.markdown("**🧑‍🌾 Sommelier & Enology Insights:**")
        insights = []

        if alcohol > 11.5:
            insights.append("✅ **High Alcohol Content (>11.5%)**: Strongly correlates with richer body and higher quality ratings in red wines.")
        elif alcohol < 9.5:
            insights.append("⚠️ **Low Alcohol Content (<9.5%)**: May result in thin body and weaker structure.")

        if volatile_acidity > 0.65:
            insights.append("❌ **High Volatile Acidity (>0.65 g/dm³)**: Risk of vinegar-like unpleasant sharpness. Lowering VA improves score.")
        else:
            insights.append("✅ **Balanced Volatile Acidity**: Below critical threshold, keeping wine fresh.")

        if sulphates > 0.65:
            insights.append("✅ **Optimal Sulphates (>0.65 g/dm³)**: Acts as strong antioxidant, preserving fruit notes.")
        else:
            insights.append("⚠️ **Low Sulphate Levels**: Sub-optimal protection against oxidation.")

        if citric_acid < 0.1:
            insights.append("💡 **Tip**: Adding slight citric acid can boost freshness and balance heavy red wines.")

        for ins in insights:
            st.markdown(f"- {ins}")

# =========================================================
# TAB 2: EXPLORATORY DATA ANALYSIS (EDA)
# =========================================================
with tab2:
    st.subheader("📈 Exploratory Data Analysis & Visualizations")
    st.markdown("Explore distribution statistics and correlations across the 1,599 red wine samples in the UCI dataset.")

    eda_col1, eda_col2 = st.columns([1, 1])

    with eda_col1:
        st.markdown("#### 📊 Dataset Quick Summary")
        st.dataframe(df.describe().T[['mean', 'std', 'min', '50%', 'max']], use_container_width=True)

        st.markdown("#### 🍷 Quality Score Count Distribution")
        fig_hist = px.histogram(
            df, x='quality', color='quality',
            color_discrete_sequence=px.colors.sequential.Plasma,
            title="Distribution of Wine Quality Ratings (3 to 8)"
        )
        fig_hist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig_hist, use_container_width=True)

    with eda_col2:
        st.markdown("#### 🔥 Feature Correlation Heatmap")
        corr = df.corr()
        fig_corr = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            color_continuous_scale="RdBu_r",
            labels=dict(color="Correlation")
        )
        fig_corr.update_layout(
            height=450,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="white", size=10)
        )
        st.plotly_chart(fig_corr, use_container_width=True)

    st.markdown("---")
    st.markdown("#### 🔍 Interactive Feature Relationship Inspector")

    inspect_col1, inspect_col2 = st.columns(2)

    with inspect_col1:
        feature_choice = st.selectbox("Select Feature to Compare against Quality:", feature_names, index=10)
        fig_box = px.box(
            df, x='quality', y=feature_choice, color='quality',
            title=f"{feature_choice.title()} vs. Quality Score",
            points="outliers",
            color_discrete_sequence=px.colors.qualitative.Vivid
        )
        fig_box.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig_box, use_container_width=True)

    with inspect_col2:
        x_axis = st.selectbox("Scatter Plot X-Axis:", feature_names, index=10) # Alcohol
        y_axis = st.selectbox("Scatter Plot Y-Axis:", feature_names, index=1)  # Volatile acidity
        fig_scatter = px.scatter(
            df, x=x_axis, y=y_axis, color='quality',
            title=f"{x_axis.title()} vs {y_axis.title()} by Quality",
            color_continuous_scale='Viridis',
            hover_data=feature_names
        )
        fig_scatter.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig_scatter, use_container_width=True)

# =========================================================
# TAB 3: MODEL EVALUATION & RETRAINING
# =========================================================
with tab3:
    st.subheader("🤖 Machine Learning Model Evaluation & Custom Retraining")

    # Current Model Metrics
    st.markdown("### 🏆 Active Model Performance Metrics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Algorithm", saved_metrics.get('best_algorithm', 'Random Forest'))
    m2.metric("Test Accuracy", f"{saved_metrics.get('accuracy', 0):.2%}")
    m3.metric("Weighted F1-Score", f"{saved_metrics.get('f1_score', 0):.4f}")
    m4.metric("Weighted Precision", f"{saved_metrics.get('precision', 0):.4f}")

    col_eval1, col_eval2 = st.columns(2)

    with col_eval1:
        st.markdown("#### 🌲 Feature Importance Ranking")
        imp_df = pd.DataFrame({
            'Feature': list(saved_importances.keys()),
            'Importance': list(saved_importances.values())
        }).sort_values('Importance', ascending=True)

        fig_imp = px.bar(
            imp_df, x='Importance', y='Feature', orientation='h',
            title="Relative Feature Importance in Quality Prediction",
            color='Importance', color_continuous_scale='Burg'
        )
        fig_imp.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
        st.plotly_chart(fig_imp, use_container_width=True)

    with col_eval2:
        st.markdown("#### 🧩 Confusion Matrix")
        cm = np.array(saved_metrics.get('confusion_matrix', []))
        classes_str = [str(c) for c in saved_metrics.get('classes', [3,4,5,6,7,8])]
        
        if cm.size > 0:
            fig_cm = px.imshow(
                cm, text_auto=True,
                x=classes_str, y=classes_str,
                labels=dict(x="Predicted Quality", y="Actual Quality"),
                color_continuous_scale="Purples"
            )
            fig_cm.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
            st.plotly_chart(fig_cm, use_container_width=True)

    st.markdown("---")
    st.markdown("### ⚙️ Interactive Model Retraining Sandbox")
    st.markdown("Experiment with different machine learning algorithms and hyperparameters:")

    retrain_col1, retrain_col2 = st.columns(2)

    with retrain_col1:
        selected_algo = st.selectbox(
            "Select Machine Learning Algorithm:",
            ["Random Forest", "Extra Trees", "Gradient Boosting", "Decision Tree", "Support Vector Machine (SVM)"]
        )
        test_size = st.slider("Test Split Proportion:", 0.10, 0.40, 0.20, 0.05)

    with retrain_col2:
        if selected_algo in ["Random Forest", "Extra Trees", "Gradient Boosting"]:
            n_estimators = st.slider("Number of Estimators (Trees):", 50, 300, 150, 25)
            max_depth = st.slider("Max Tree Depth:", 3, 20, 12, 1)
        elif selected_algo == "Decision Tree":
            max_depth = st.slider("Max Tree Depth:", 2, 20, 8, 1)
        elif selected_algo == "Support Vector Machine (SVM)":
            c_val = st.select_slider("Regularization Parameter (C):", [0.1, 1.0, 10.0, 100.0], value=1.0)

    if st.button("🚀 Train & Benchmark Selected Model", type="primary"):
        with st.spinner("Training model on dataset..."):
            X = df[feature_names]
            y = df['quality']

            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
            sc = StandardScaler()
            X_tr_sc = sc.fit_transform(X_tr)
            X_te_sc = sc.transform(X_te)

            if selected_algo == "Random Forest":
                clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            elif selected_algo == "Extra Trees":
                clf = ExtraTreesClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            elif selected_algo == "Gradient Boosting":
                clf = GradientBoostingClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
            elif selected_algo == "Decision Tree":
                clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
            else:
                clf = SVC(C=c_val, random_state=42)

            clf.fit(X_tr_sc, y_tr)
            preds = clf.predict(X_te_sc)

            acc = accuracy_score(y_te, preds)
            f1 = f1_score(y_te, preds, average='weighted', zero_division=0)
            prec = precision_score(y_te, preds, average='weighted', zero_division=0)

            st.success(f"Training Complete! **{selected_algo}** achieved **{acc:.2%} Accuracy** (F1: {f1:.4f}, Precision: {prec:.4f})")

# =========================================================
# TAB 4: BATCH PREDICTION
# =========================================================
with tab4:
    st.subheader("📁 Batch Wine Sample Quality Prediction")
    st.markdown("Upload a CSV file containing multiple wine chemical measurement rows to obtain predictions in bulk.")

    col_batch1, col_batch2 = st.columns([2, 1])

    with col_batch2:
        st.markdown("#### 📥 Sample Template")
        st.markdown("Download a sample CSV format pre-filled with test samples:")
        sample_batch = df.head(5)[feature_names]
        csv_sample = sample_batch.to_csv(index=False)
        st.download_button(
            label="Download CSV Template",
            data=csv_sample,
            file_name="wine_quality_input_template.csv",
            mime="text/csv"
        )

    with col_batch1:
        uploaded_file = st.file_uploader("Upload Wine Samples CSV file:", type=["csv"])

    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            missing_cols = [col for col in feature_names if col not in batch_df.columns]

            if missing_cols:
                st.error(f"Missing required columns in uploaded CSV: `{missing_cols}`")
            else:
                st.success(f"Successfully loaded {len(batch_df)} wine samples.")
                
                # Scale & Predict
                batch_scaled = scaler.transform(batch_df[feature_names])
                batch_preds = model.predict(batch_scaled)
                
                batch_df['Predicted_Quality'] = batch_preds
                
                def map_cat(q):
                    if q >= 7: return "High"
                    elif q >= 5: return "Medium"
                    else: return "Low"
                    
                batch_df['Quality_Category'] = [map_cat(q) for q in batch_preds]

                st.markdown("#### 📊 Prediction Results:")
                st.dataframe(batch_df[['Predicted_Quality', 'Quality_Category'] + feature_names], use_container_width=True)

                # Distribution summary
                fig_batch_dist = px.histogram(
                    batch_df, x='Predicted_Quality', color='Quality_Category',
                    title="Batch Predicted Quality Ratings Distribution",
                    color_discrete_map={'High': '#2d6a4f', 'Medium': '#9c6644', 'Low': '#800f2f'}
                )
                fig_batch_dist.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="white"))
                st.plotly_chart(fig_batch_dist, use_container_width=True)

                # Download Results Button
                result_csv = batch_df.to_csv(index=False)
                st.download_button(
                    label="💾 Download Predicted Results CSV",
                    data=result_csv,
                    file_name="wine_quality_predictions.csv",
                    mime="text/csv"
                )
        except Exception as e:
            st.error(f"Error processing file: {e}")

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #888888;'>Wine Quality AI Predictor | Powered by Streamlit & Scikit-Learn</div>",
    unsafe_allow_html=True
)
