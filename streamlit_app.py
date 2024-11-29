import streamlit as st
import pandas as pd
import numpy as np
import random
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import shap
from lime.lime_tabular import LimeTabularExplainer
from faker import Faker
import warnings
import xgboost as xgb

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Initialize Faker
fake = Faker('it_IT')

# Function to generate Codice Fiscale (Italian tax code)
def generate_codice_fiscale():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=16))

# Function to generate synthetic data
def generate_data(num_samples):
    regions = ['Lombardia', 'Lazio', 'Campania', 'Sicilia', 'Veneto', 'Piemonte',
               'Toscana', 'Emilia-Romagna', 'Puglia', 'Calabria']
    provinces = {
        'Lombardia': ['Milano', 'Bergamo', 'Brescia'],
        'Lazio': ['Roma', 'Frosinone', 'Latina'],
        'Campania': ['Napoli', 'Salerno', 'Caserta'],
        'Sicilia': ['Palermo', 'Catania', 'Messina'],
        'Veneto': ['Venezia', 'Verona', 'Padova'],
        'Piemonte': ['Torino', 'Novara', 'Cuneo'],
        'Toscana': ['Firenze', 'Siena', 'Pisa'],
        'Emilia-Romagna': ['Bologna', 'Modena', 'Parma'],
        'Puglia': ['Bari', 'Lecce', 'Taranto'],
        'Calabria': ['Catanzaro', 'Reggio Calabria', 'Cosenza']
    }
    genders = ['M', 'F']
    professions = ['Ingegnere', 'Medico', 'Avvocato', 'Commercialista', 'Manager',
                   'Imprenditore', 'Artista', 'Scienziato', 'Disoccupato']
    segments = ['Retail', 'Corporate']
    risk_levels = ['Basso', 'Medio', 'Alto']

    data = {
        'codice_fiscale': [generate_codice_fiscale() for _ in range(num_samples)],
        'età': np.random.randint(18, 90, size=num_samples),
        'patrimonio': np.round(np.random.lognormal(mean=10, sigma=1, size=num_samples), 2),
        'numero_operazioni': np.random.poisson(lam=100, size=num_samples),
        'sesso': np.random.choice(genders, size=num_samples),
        'regione': np.random.choice(regions, size=num_samples),
        'provincia': [random.choice(provinces[region]) for region in np.random.choice(regions, size=num_samples)],
        'città': [fake.city() for _ in range(num_samples)],
        'note': np.random.choice(['Attività regolare', 'Pattern di transazioni insoliti',
                                   'Trasferimenti internazionali'], size=num_samples, p=[0.7, 0.2, 0.1]),
        'rischio': np.random.choice(risk_levels, size=num_samples, p=[0.6, 0.3, 0.1]),
        'segmento': np.random.choice(segments, size=num_samples, p=[0.7, 0.3]),
        'entrate_annuali': np.round(np.random.normal(loc=100000, scale=30000, size=num_samples), 2),
        'occupazione': np.random.choice(professions, size=num_samples),
        'pep': np.random.choice([True, False], size=num_samples, p=[0.05, 0.95]),
        'transazioni_anomale': np.random.choice([1, 0], size=num_samples, p=[0.1, 0.9])
    }

    df = pd.DataFrame(data)
    df['entrate_annuali'] = df['entrate_annuali'].apply(lambda x: x if x > 0 else 20000)
    df['patrimonio'] = df['patrimonio'].apply(lambda x: x if x > 0 else 10000)
    return df

# Load data with caching
@st.cache_data
def load_data(num_samples=1000):
    return generate_data(num_samples)

# Updated preprocess_data function
def preprocess_data(df, target_column, selected_features=None):
    """
    Preprocess the dataset by dropping unnecessary columns and one-hot encoding categorical variables.

    Args:
        df (pd.DataFrame): The input dataframe.
        target_column (str): The name of the target column.
        selected_features (list or None): List of features to include, or None to include all available features.

    Returns:
        X (pd.DataFrame): Processed features.
        y (pd.Series): Target variable.
    """
    # If no features are selected, use all columns except the target and known exclusions
    if selected_features is None:
        selected_features = df.columns.tolist()
        selected_features = list(set(selected_features) - set(['codice_fiscale', 'note', 'città', target_column]))

    # Drop columns not in selected features
    drop_cols = set(df.columns) - set(selected_features) - {target_column}
    features = df.drop(columns=drop_cols)
    
    # One-hot encode categorical variables
    X = pd.get_dummies(features, drop_first=True)
    y = df[target_column].astype(int)
    
    return X, y

# Function to train models
def train_models(X_train, y_train):
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    for name, model in models.items():
        model.fit(X_train, y_train)

    estimators = [
        ('rf', models['Random Forest']),
        ('gb', models['Gradient Boosting']),
        ('xgb', models['XGBoost'])
    ]
    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(),
        passthrough=True,
        cv=5
    )
    stacking_model.fit(X_train, y_train)
    models['Stacking Classifier'] = stacking_model

    return models

# Main Streamlit App
def main():
    st.set_page_config(page_title="AML Detection - SGR", layout="wide")

    st.sidebar.title("AML Detection Dashboard")
    tabs = ["Data Overview", "Variable Selection", "Model Training", "Model Evaluation", "Explainability"]
    selected_tab = st.sidebar.radio("Navigate to", tabs)

    # Load the dataset
    df = load_data(1000)
    target = 'transazioni_anomale'

    if selected_tab == "Variable Selection":
        st.title("Variable Selection")
        X_all, y = preprocess_data(df, target)  # Preprocess data without variable selection
        selected_features = st.sidebar.multiselect(
            "Select Features to Include",
            options=X_all.columns.tolist(),
            default=X_all.columns.tolist()
        )
        st.session_state['selected_features'] = selected_features

        st.write(f"Selected {len(selected_features)} features.")
        st.dataframe(selected_features)

    elif selected_tab == "Model Training":
        st.title("Model Training")
        selected_features = st.session_state.get('selected_features', None)
        X, y = preprocess_data(df, target, selected_features)

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        st.write(f"Training Samples: {X_train.shape[0]}")
        st.write(f"Testing Samples: {X_test.shape[0]}")

        with st.spinner("Training models..."):
            models = train_models(X_train, y_train)
        st.success("Models trained successfully!")
        st.session_state['models'] = models
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test

    # Add other tabs for Model Evaluation and Explainability here as needed

if __name__ == "__main__":
    main()