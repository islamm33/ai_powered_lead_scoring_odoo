# custom_addons/lead_scoring_ai/models/train_pipeline.py
import pandas as pd
import joblib
import sys
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from datetime import datetime
# Import transformers
from odoo.addons.lead_scoring_ai.models.job_position_transformer import JobPositionTransformer
from odoo.addons.lead_scoring_ai.models.lead_preprocessor import LeadPreprocessor

# Configuration
MODE = "train"  # Change to "inference" when needed
DATA_PATH = r'C:\Program Files\Odoo 18.0.20250413\custom_addons\lead_scoring_ai\data'
PIPELINE_PATH = f'{DATA_PATH}\\trained_pipelinex.joblib'
ENCODER_PATH = f'{DATA_PATH}\\lead_encoderx.joblib'

# Common column configuration
SELECTED_COLUMNS = [
    'Lead Source', 'Do Not Email', 'Do Not Call', 'Country',
    'City', 'Specialization', 'Through Recommendations',
    'Last Notable Activity', 'Tags'
]


def run_training():
    """Train and save the complete pipeline"""
    # Load and prepare training data
    df = pd.read_csv(f'{DATA_PATH}\\Lead Scoring.csv')
    X = df[SELECTED_COLUMNS]
    y = df['Converted']

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Configure pipeline with training-mode settings
    pipeline = Pipeline(steps=[
        ('job_position_processing', JobPositionTransformer(mode="train")),
        ('preprocessing', LeadPreprocessor(encoder_path=ENCODER_PATH)),
        ('classifier', RandomForestClassifier(n_estimators=500, random_state=42))
    ])

    # Train and evaluate
    pipeline.fit(X_train, y_train)
    score = pipeline.score(X_test, y_test)
    print(f"Model trained successfully. Test accuracy: {score:.2f}")

    # In run_training():
    pipeline_data = {
        'pipeline': pipeline,
        'selected_columns': SELECTED_COLUMNS,
        'categorical_columns': pipeline.named_steps['preprocessing'].categorical_columns,
        'version': datetime.now().strftime('v%Y%m%d')
    }
    joblib.dump(pipeline_data, PIPELINE_PATH)
    print(f"Pipeline saved to {PIPELINE_PATH}")
    return score


def run_inference():
    """Load saved pipeline and score new leads"""
    # Load trained pipeline
    try:
        pipeline = joblib.load(PIPELINE_PATH)
    except FileNotFoundError:
        print("Error: No trained pipeline found. Train first!")
        sys.exit(1)

    # Load new data (example format)
    new_leads = pd.read_csv(f'{DATA_PATH}\\new_leads.csv')[SELECTED_COLUMNS]

    # Predict probabilities
    probabilities = pipeline.predict_proba(new_leads)[:, 1]  # Get positive class probabilities

    # Format results
    results = pd.DataFrame({
        'lead_id': new_leads.index,
        'conversion_probability': probabilities
    })

    print("Inference results:")
    print(results.head())

    # Optional: Save predictions
    results.to_csv(f'{DATA_PATH}\\predictions.csv', index=False)
    print(f"Predictions saved to {DATA_PATH}\\predictions.csv")


if __name__ == "__main__":
    if MODE == "train":
        run_training()
    elif MODE == "inference":
        run_inference()
    else:
        print(f"Invalid mode: {MODE}. Use 'train' or 'inference'")
        sys.exit(1)