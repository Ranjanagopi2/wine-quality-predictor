import os
import urllib.request
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
import joblib

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
DATA_PATH = os.path.join(DATA_DIR, 'winequality-red.csv')
MODEL_PATH = os.path.join(MODEL_DIR, 'wine_quality_model.joblib')

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

def load_or_fetch_data():
    """Download UCI Red Wine Quality dataset or generate fallback data based on UCI stats."""
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    if os.path.exists(DATA_PATH):
        print(f"Loading cached dataset from {DATA_PATH}...")
        df = pd.read_csv(DATA_PATH, sep=';' if ';' in open(DATA_PATH).readline() else ',')
        return df

    print("Attempting to download dataset from UCI repository...")
    try:
        df = pd.read_csv(url, sep=';')
        df.to_csv(DATA_PATH, index=False)
        print(f"Dataset downloaded successfully and saved to {DATA_PATH}.")
        return df
    except Exception as e:
        print(f"Download failed ({e}). Generating realistic synthetic dataset matching UCI distributions...")
        np.random.seed(42)
        n_samples = 1599
        
        # Distributions based on real UCI Red Wine dataset stats
        fixed_acidity = np.random.normal(8.32, 1.74, n_samples).clip(4.6, 15.9)
        volatile_acidity = np.random.normal(0.53, 0.18, n_samples).clip(0.12, 1.58)
        citric_acid = np.random.normal(0.27, 0.19, n_samples).clip(0.0, 1.0)
        residual_sugar = np.random.exponential(2.5, n_samples).clip(0.9, 15.5)
        chlorides = np.random.normal(0.087, 0.047, n_samples).clip(0.012, 0.611)
        free_sulfur_dioxide = np.random.exponential(14.0, n_samples).clip(1.0, 72.0)
        total_sulfur_dioxide = np.random.normal(46.5, 32.9, n_samples).clip(6.0, 289.0)
        density = np.random.normal(0.9967, 0.0019, n_samples).clip(0.990, 1.0037)
        pH = np.random.normal(3.31, 0.15, n_samples).clip(2.74, 4.01)
        sulphates = np.random.normal(0.66, 0.17, n_samples).clip(0.33, 2.0)
        alcohol = np.random.normal(10.42, 1.07, n_samples).clip(8.4, 14.9)

        # Realistic formula for quality based on known feature effects
        # Alcohol(+), Sulphates(+), Citric Acid(+), Volatile Acidity(-), Total SO2(-)
        score = (
            0.45 * alcohol +
            1.2 * sulphates +
            0.8 * citric_acid -
            2.2 * volatile_acidity -
            0.01 * total_sulfur_dioxide +
            0.1 * fixed_acidity -
            10 * (density - 0.996) +
            np.random.normal(0, 0.6, n_samples)
        )

        quality = np.digitize(score, bins=np.percentile(score, [5, 20, 60, 88, 97])) + 3
        quality = np.clip(quality, 3, 8)

        df = pd.DataFrame({
            'fixed acidity': np.round(fixed_acidity, 2),
            'volatile acidity': np.round(volatile_acidity, 3),
            'citric acid': np.round(citric_acid, 3),
            'residual sugar': np.round(residual_sugar, 2),
            'chlorides': np.round(chlorides, 4),
            'free sulfur dioxide': np.round(free_sulfur_dioxide, 1),
            'total sulfur dioxide': np.round(total_sulfur_dioxide, 1),
            'density': np.round(density, 4),
            'pH': np.round(pH, 2),
            'sulphates': np.round(sulphates, 2),
            'alcohol': np.round(alcohol, 2),
            'quality': quality.astype(int)
        })

        df.to_csv(DATA_PATH, index=False)
        print(f"Generated dataset saved to {DATA_PATH}.")
        return df

def train_and_save():
    df = load_or_fetch_data()
    print(f"Dataset Shape: {df.shape}")
    print(f"Quality distribution:\n{df['quality'].value_counts().sort_index()}")

    feature_cols = [c for c in df.columns if c != 'quality']
    X = df[feature_cols]
    y = df['quality']

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Random Forest
    rf_model = RandomForestClassifier(n_estimators=150, max_depth=12, random_state=42)
    rf_model.fit(X_train_scaled, y_train)

    # Train Extra Trees
    et_model = ExtraTreesClassifier(n_estimators=150, random_state=42)
    et_model.fit(X_train_scaled, y_train)

    rf_acc = accuracy_score(y_test, rf_model.predict(X_test_scaled))
    et_acc = accuracy_score(y_test, et_model.predict(X_test_scaled))

    print(f"Random Forest Test Accuracy: {rf_acc:.4f}")
    print(f"Extra Trees Test Accuracy:   {et_acc:.4f}")

    best_model = rf_model if rf_acc >= et_acc else et_model
    best_name = "Random Forest" if rf_acc >= et_acc else "Extra Trees"
    y_pred = best_model.predict(X_test_scaled)

    metrics = {
        'best_algorithm': best_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
        'classes': best_model.classes_.tolist()
    }

    feature_importances = pd.Series(
        best_model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False).to_dict()

    save_payload = {
        'model': best_model,
        'scaler': scaler,
        'feature_names': feature_cols,
        'metrics': metrics,
        'feature_importances': feature_importances,
        'classes': best_model.classes_
    }

    joblib.dump(save_payload, MODEL_PATH)
    print(f"Model and artifacts successfully saved to {MODEL_PATH}")

if __name__ == '__main__':
    train_and_save()
