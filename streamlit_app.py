import streamlit as st
import pandas as pd
import numpy as np
import random
import string
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve
)
import shap
from lime.lime_tabular import LimeTabularExplainer
from sklearn.preprocessing import LabelEncoder
from faker import Faker
import warnings
import xgboost as xgb
import logging

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Initialize Faker for generating fake city names
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

# Initialize logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Define a custom handler that appends to a list in session state
class StreamlitHandler(logging.Handler):
    def emit(self, record):
        log_entry = self.format(record)
        if 'logs' not in st.session_state:
            st.session_state['logs'] = []
        st.session_state['logs'].append(log_entry)

# Add the custom handler to the logger
streamlit_handler = StreamlitHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
streamlit_handler.setFormatter(formatter)
logger.addHandler(streamlit_handler)

# Load data with caching
@st.cache_data
def load_data(uploaded_file, num_samples=1000):
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
                logger.info(f"Loaded data from CSV file: {uploaded_file.name}")
            elif uploaded_file.name.endswith(('.xls', '.xlsx')):
                df = pd.read_excel(uploaded_file)
                logger.info(f"Loaded data from Excel file: {uploaded_file.name}")
            else:
                st.error("Unsupported file format. Please upload a CSV or Excel file.")
                df = generate_data(num_samples)
                logger.warning("Unsupported file format uploaded. Generated synthetic data.")
        except Exception as e:
            st.error(f"Error loading data: {e}")
            df = generate_data(num_samples)
            logger.error(f"Error loading data: {e}. Generated synthetic data.")
    else:
        df = generate_data(num_samples)
        logger.info("No file uploaded. Generated synthetic data.")
    return df

# Preprocess data function with Label Encoding
def preprocess_data(df, target_column, selected_features=None):
    """
    Preprocess the dataset by dropping unnecessary columns and applying Label Encoding to categorical variables.

    Args:
        df (pd.DataFrame): The input dataframe.
        target_column (str): The name of the target column.
        selected_features (list or None): List of features to include, or None to include all available features.

    Returns:
        X (pd.DataFrame): Processed features.
        y (pd.Series): Target variable.
    """
    logger.info("Starting data preprocessing.")
    # If no features are selected, use all columns except the target and known exclusions
    if selected_features is None:
        selected_features = df.columns.tolist()
        selected_features = list(set(selected_features) - set(['codice_fiscale', 'note', 'città', target_column]))
        logger.info("No selected features provided. Using all available features except exclusions.")

    # Ensure selected_features are in the dataframe
    selected_features = [feature for feature in selected_features if feature in df.columns]
    logger.info(f"Selected features: {selected_features}")

    # Drop columns not in selected features
    drop_cols = set(df.columns) - set(selected_features) - {target_column}
    features = df.drop(columns=drop_cols)
    logger.info(f"Dropped columns: {drop_cols}")

    # Identify categorical columns
    categorical_cols = features.select_dtypes(include=['object', 'bool']).columns.tolist()
    logger.info(f"Categorical columns identified for Label Encoding: {categorical_cols}")

    # Apply Label Encoding to categorical columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        features[col] = le.fit_transform(features[col])
        label_encoders[col] = le  # Store the encoder if needed later
        logger.info(f"Applied Label Encoding to column: {col}")

    X = features
    y = df[target_column].astype(int)
    logger.info("Data preprocessing completed.")
    return X, y

# Function to train models
def train_models(X_train, y_train):
    logger.info("Starting training of models...")
    models = {
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
        'XGBoost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    }

    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        logger.info(f"{name} trained successfully.")

    # Stacking Classifier
    logger.info("Training Stacking Classifier...")
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
    logger.info("Stacking Classifier trained successfully.")
    models['Stacking Classifier'] = stacking_model

    logger.info("All models trained successfully.")
    return models

# Function to evaluate models
def evaluate_models(models, X_test, y_test):
    logger.info("Starting evaluation of models.")
    evaluation = {}
    for name, model in models.items():
        logger.info(f"Evaluating {name}...")
        y_pred = model.predict(X_test)
        if hasattr(model, "predict_proba"):
            y_proba = model.predict_proba(X_test)[:,1]
        else:
            y_proba = model.decision_function(X_test)
            y_proba = (y_proba - y_proba.min()) / (y_proba.max() - y_proba.min())  # Normalize
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        fpr, tpr, thresholds = roc_curve(y_test, y_proba)
        evaluation[name] = {
            'classification_report': report,
            'confusion_matrix': cm,
            'roc_auc': roc_auc,
            'fpr': fpr,
            'tpr': tpr,
            'thresholds': thresholds
        }
        logger.info(f"{name} evaluated successfully.")
    logger.info("All models evaluated successfully.")
    return evaluation

# Main Streamlit App
def main():
    st.set_page_config(page_title="AML Detection - SGR", layout="wide")

    st.sidebar.title("AML Detection Dashboard")
    tabs = ["Data Overview", "Transformation and Selection", "Model Training", "Model Evaluation", "Explainability"]
    selected_tab = st.sidebar.radio("Navigate to", tabs)

    # File uploader
    st.sidebar.subheader("Upload Your Data")
    uploaded_file = st.sidebar.file_uploader("Upload Excel or CSV", type=['csv', 'xlsx', 'xls'])

    # Load the dataset
    df = load_data(uploaded_file, num_samples=1000)
    target = 'transazioni_anomale'

    # Display data upload info
    if uploaded_file is not None:
        st.sidebar.success(f"Uploaded file: {uploaded_file.name}")
    else:
        st.sidebar.info("No file uploaded. Using generated data.")

    # Initialize logs in session state
    if 'logs' not in st.session_state:
        st.session_state['logs'] = []

    # Transformation and Selection
    if selected_tab == "Transformation and Selection":
        st.title("Transformation and Selection")

        # Preprocess data without variable selection to get all features
        X_all, y = preprocess_data(df, target)

        # Multiselect for feature selection
        selected_features = st.multiselect(
            "Select Features to Include",
            options=X_all.columns.tolist(),
            default=X_all.columns.tolist()
        )

        if not selected_features:
            st.warning("Please select at least one feature.")
        else:
            st.success(f"Selected {len(selected_features)} features.")
            st.session_state['selected_features'] = selected_features

            # Show selected features
            st.subheader("Selected Features")
            st.write(selected_features)

            # Apply Feature Importance using XGBoost to suggest relevant features
            st.subheader("Feature Importance using XGBoost")
            if 'models' in st.session_state and 'XGBoost' in st.session_state['models']:
                model = st.session_state['models']['XGBoost']
                importances = model.feature_importances_
                feature_importance_df = pd.DataFrame({
                    'Feature': X_all.columns,
                    'Importance': importances
                }).sort_values(by='Importance', ascending=False)

                st.write(feature_importance_df)

                # Plot Feature Importances
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.barplot(x='Importance', y='Feature', data=feature_importance_df.head(20), ax=ax)
                ax.set_title('Top 20 Feature Importances')
                st.pyplot(fig)
            else:
                st.info("Train the models first to view feature importances.")

    elif selected_tab == "Data Overview":
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
        missing_data = missing_data[missing_data > 0]
        if not missing_data.empty:
            fig, ax = plt.subplots()
            sns.barplot(x=missing_data.index, y=missing_data.values, ax=ax)
            ax.set_ylabel("Number of Missing Values")
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
            st.pyplot(fig)
        else:
            st.write("No missing data found.")

        # Correlation Heatmap
        st.subheader("Correlation Heatmap")
        numeric_df = df.select_dtypes(include=[np.number])
        if not numeric_df.empty:
            fig, ax = plt.subplots(figsize=(12, 8))
            sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)
        else:
            st.write("No numeric features available for correlation heatmap.")

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

        # Get selected features
        selected_features = st.session_state.get('selected_features', None)
        if selected_features is None:
            # If transformation and selection not done, use all features except exclusions
            selected_features = list(set(df.columns.tolist()) - set(['codice_fiscale', 'note', 'città', target]))
            st.session_state['selected_features'] = selected_features
            st.info("Using all features as no transformation and selection was performed.")

        # Preprocess data with selected features
        X, y = preprocess_data(df, target, selected_features)

        st.subheader("Data Preview")
        st.write("Features:")
        st.dataframe(X.head())

        st.write("Target:")
        st.dataframe(y.head())

        # Sidebar for training parameters
        st.sidebar.subheader("Training Parameters")
        test_size = st.sidebar.slider("Test Set Size (%)", 10, 50, 20, step=5)
        random_state = st.sidebar.number_input("Random State", value=42, step=1)
        n_samples = st.sidebar.number_input("Number of Samples", min_value=100, max_value=10000, value=1000, step=100)

        # Update dataset size if user changes
        if 'original_num_samples' not in st.session_state:
            st.session_state['original_num_samples'] = len(df)

        if n_samples != st.session_state.get('original_num_samples', 1000):
            df = generate_data(int(n_samples))
            st.session_state['original_num_samples'] = n_samples
            st.sidebar.info(f"Data regenerated with {n_samples} samples.")
            X, y = preprocess_data(df, target, selected_features)
            st.session_state['selected_features'] = selected_features
            logger.info(f"Data regenerated with {n_samples} samples.")

        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size/100, random_state=random_state, stratify=y
        )

        st.write(f"Training Samples: {X_train.shape[0]}")
        st.write(f"Testing Samples: {X_test.shape[0]}")

        # Initialize logs
        st.session_state['logs'] = []
        log_placeholder = st.empty()

        # Function to display logs
        def display_logs():
            with log_placeholder.container():
                st.text_area("Logs", value="\n".join(st.session_state['logs']), height=300)

        # Train models and capture logs
        with st.spinner("Training models..."):
            models = train_models(X_train, y_train)
            st.success("Models trained successfully!")
            st.session_state['models'] = models
            st.session_state['X_test'] = X_test
            st.session_state['y_test'] = y_test
            st.session_state['X_train'] = X_train
            st.session_state['y_train'] = y_train

        # Display logs
        display_logs()

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

        # Performance Comparison Table
        st.subheader("Performance Comparison")
        performance_data = []
        for name, metrics in evaluation.items():
            performance_data.append({
                'Model': name,
                'ROC AUC': metrics['roc_auc'],
                'Precision': metrics['classification_report']['1']['precision'],
                'Recall': metrics['classification_report']['1']['recall'],
                'F1-Score': metrics['classification_report']['1']['f1-score']
            })
        performance_df = pd.DataFrame(performance_data)
        performance_df.set_index('Model', inplace=True)
        st.dataframe(performance_df)

        # ROC Curves
        st.subheader("ROC Curves")
        plt.figure(figsize=(10, 8))
        for name, metrics in evaluation.items():
            plt.plot(metrics['fpr'], metrics['tpr'], label=f'{name} (AUC = {metrics["roc_auc"]:.2f})')
        plt.plot([0,1], [0,1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curves Comparison')
        plt.legend(loc='lower right')
        st.pyplot(plt)

        # Detailed Metrics per Model
        for name, metrics in evaluation.items():
            st.subheader(f"{name}")
            st.write(f"**ROC AUC Score:** {metrics['roc_auc']:.4f}")

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
        X_train = st.session_state.get('X_train', None)

        # Select model for explanation
        model_name = st.selectbox("Select Model for Explanation", list(models.keys()))
        model = models[model_name]

        # SHAP Explanation
        st.subheader("SHAP Explanation")
        with st.spinner("Generating SHAP values..."):
            try:
                if isinstance(model, (RandomForestClassifier, GradientBoostingClassifier, xgb.XGBClassifier, StackingClassifier)):
                    explainer = shap.Explainer(model, X_train)
                elif isinstance(model, LogisticRegression):
                    explainer = shap.LinearExplainer(model, X_train, feature_dependence="independent")
                else:
                    explainer = shap.Explainer(model, X_train)
                shap_values = explainer(X_test)
                plt.figure(figsize=(10, 6))
                shap.plots.beeswarm(shap_values, show=False)
                st.pyplot(plt)
            except Exception as e:
                st.error(f"SHAP Error: {e}")

        # LIME Explanation
        st.subheader("LIME Explanation")
        with st.spinner("Generating LIME explanation..."):
            try:
                # Ensure that X_train is available
                if X_train is None:
                    st.error("Training data not available for LIME explanations.")
                else:
                    explainer_lime = LimeTabularExplainer(
                        X_train.values,
                        feature_names=X_train.columns.tolist(),
                        class_names=['No', 'Yes'],
                        mode='classification'
                    )

                    idx = st.slider('Select Index for LIME Explanation', 0, len(X_test) - 1, 0)
                    exp = explainer_lime.explain_instance(X_test.iloc[idx].values, model.predict_proba, num_features=10)
                    fig, ax = plt.subplots(figsize=(10, 5))
                    exp.as_pyplot_figure(ax)
                    st.pyplot(fig)

                    st.write("### LIME Explanation Details")
                    st.write(exp.as_list())
            except Exception as e:
                st.error(f"LIME Error: {e}")

        # Additional SHAP Plots (Optional)
        st.subheader("SHAP Summary Plot")
        with st.spinner("Generating SHAP summary plot..."):
            try:
                plt.figure(figsize=(10, 6))
                shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
                st.pyplot(plt)
            except Exception as e:
                st.error(f"SHAP Summary Plot Error: {e}")

    if __name__ == "__main__":
        main()
