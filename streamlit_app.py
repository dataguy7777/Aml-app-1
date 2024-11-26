import streamlit as st
import pandas as pd
import numpy as np
import random
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import shap
from lime.lime_tabular import LimeTabularExplainer
from datetime import datetime, timedelta
from faker import Faker
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Initialize Faker
fake = Faker('it_IT')

# Function to generate Codice Fiscale (Italian tax code)
def generate_codice_fiscale():
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=16))

# Function to generate data
def generate_data(num_samples):
    # List of regions, provinces, and cities
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
    
    # More realistic feature distributions
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
        'pep': np.random.choice([True, False], size=num_samples, p=[0.05, 0.95]),  # Politically Exposed Person
        'transazioni_anomale': np.random.choice([True, False], size=num_samples, p=[0.1, 0.9])  # Suspicious transactions
    }

    # Create DataFrame
    df = pd.DataFrame(data)
    
    # Handle negative or unrealistic values if any
    df['entrate_annuali'] = df['entrate_annuali'].apply(lambda x: x if x > 0 else 20000)
    df['patrimonio'] = df['patrimonio'].apply(lambda x: x if x > 0 else 10000)
    
    return df

# Load data with caching
@st.cache_data
def load_data(num_samples=1000):
    return generate_data(num_samples)

# Function to preprocess data
def preprocess_data(df, target_column):
    features = df.drop(columns=[target_column, 'codice_fiscale', 'note', 'città'])
    X = pd.get_dummies(features, drop_first=True)  # One-hot encoding
    y = df[target_column].astype(int)
    return X, y

# Function to train models
def train_models(X_train, y_train):
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
    }
    
    for name, model in models.items():
        model.fit(X_train, y_train)
    
    return models

# Function to evaluate models
def evaluate_models(models, X_test, y_test):
    evaluation = {}
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:,1]
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        evaluation[name] = {
            'classification_report': report,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'fpr': None,
            'tpr': None,
            'thresholds': None
        }
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        evaluation[name]['fpr'] = fpr
        evaluation[name]['tpr'] = tpr
        evaluation[name]['thresholds'] = thresholds
    return evaluation

# Main Streamlit App
def main():
    st.set_page_config(page_title="AML Detection - SGR", layout="wide")

    # Load the dataset
    df = load_data(1000)

    # Sidebar for navigation and settings
    st.sidebar.title("AML Detection Dashboard")
    tabs = ["Data Overview", "Model Training", "Model Evaluation", "Explainability"]
    selected_tab = st.sidebar.radio("Navigate to", tabs)

    # Common preprocessing
    target = 'transazioni_anomale'
    X, y = preprocess_data(df, target)

    if selected_tab == "Data Overview":
        st.title("Data Overview")
        
        # Display raw data
        st.subheader("First 10 Records")
        st.dataframe(df.head(10))
        
        # Summary statistics
        st.subheader("Summary Statistics")
        st.write(df.describe())
        
        # Missing data
        st.subheader("Missing Data")
        missing_data = df.isnull().sum()
        st.bar_chart(missing_data[missing_data > 0])
        
        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap="coolwarm")
        st.pyplot(plt)
        
        # Distribution plots
        st.subheader("Feature Distributions")
        numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
        for feature in numeric_features:
            fig, ax = plt.subplots()
            sns.histplot(df[feature], kde=True, ax=ax)
            ax.set_title(f'Distribution of {feature}')
            st.pyplot(fig)

    elif selected_tab == "Model Training":
        st.title("Model Training")
        
        st.sidebar.subheader("Training Parameters")
        test_size = st.sidebar.slider("Test Set Size (%)", 10, 50, 20, step=5)
        random_state = st.sidebar.number_input("Random State", value=42, step=1)
        n_samples = st.sidebar.number_input("Number of Samples", min_value=100, max_value=10000, value=1000, step=100)
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=random_state, stratify=y
        )
        
        st.write(f"Training Samples: {X_train.shape[0]}")
        st.write(f"Testing Samples: {X_test.shape[0]}")
        
        # Train models
        with st.spinner("Training models..."):
            models = train_models(X_train, y_train)
        st.success("Models trained successfully!")
        
        # Store models and data in session state for later use
        st.session_state['models'] = models
        st.session_state['X_test'] = X_test
        st.session_state['y_test'] = y_test

    elif selected_tab == "Model Evaluation":
        st.title("Model Evaluation")
        
        if 'models' not in st.session_state:
            st.warning("Please train the models first in the 'Model Training' tab.")
            return
        
        models = st.session_state['models']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        
        with st.spinner("Evaluating models..."):
            evaluation = evaluate_models(models, X_test, y_test)
        st.success("Models evaluated successfully!")
        
        for name, metrics in evaluation.items():
            st.subheader(f"{name}")
            st.write(f"**ROC AUC Score:** {metrics['roc_auc']:.4f}")
            
            # ROC Curve
            fig, ax = plt.subplots()
            ax.plot(metrics['fpr'], metrics['tpr'], label=f'{name} (AUC = {metrics["roc_auc"]:.2f})')
            ax.plot([0,1], [0,1], 'k--')
            ax.set_xlabel('False Positive Rate')
            ax.set_ylabel('True Positive Rate')
            ax.set_title('ROC Curve')
            ax.legend(loc='lower right')
            st.pyplot(fig)
            
            # Confusion Matrix
            st.write("**Confusion Matrix**")
            cm = metrics['confusion_matrix']
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title('Confusion Matrix')
            st.pyplot(fig)
            
            # Classification Report
            st.write("**Classification Report**")
            report_df = pd.DataFrame(metrics['classification_report']).transpose()
            st.dataframe(report_df.style.highlight_max(axis=0))

    elif selected_tab == "Explainability":
        st.title("Model Explainability")
        
        if 'models' not in st.session_state:
            st.warning("Please train and evaluate the models first.")
            return
        
        models = st.session_state['models']
        X_test = st.session_state['X_test']
        y_test = st.session_state['y_test']
        
        # Select model for explanation
        model_name = st.selectbox("Select Model for Explanation", list(models.keys()))
        model = models[model_name]
        
        # SHAP Explanation
        st.subheader("SHAP Explanation")
        with st.spinner("Generating SHAP values..."):
            explainer = shap.Explainer(model, X_test)
            shap_values = explainer(X_test)
        st.pyplot(shap.plots.beeswarm(shap_values, show=False))
        st.pyplot()
        
        # LIME Explanation
        st.subheader("LIME Explanation")
        explainer_lime = LimeTabularExplainer(
            X_train.values if 'X_train' in locals() else X.iloc[:800].values,
            feature_names=X.columns,
            class_names=['No', 'Yes'],
            mode='classification'
        )
        
        idx = st.slider('Select Index for LIME Explanation', 0, len(X_test) - 1, 0)
        exp = explainer_lime.explain_instance(X_test.iloc[idx].values, model.predict_proba, num_features=10)
        fig, ax = plt.subplots()
        exp.as_pyplot_figure(ax)
        st.pyplot(fig)
        
        st.write("### LIME Explanation Details")
        st.write(exp.as_list())

if __name__ == "__main__":
    main()
