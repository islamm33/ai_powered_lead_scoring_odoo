from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  roc_auc_score
from sklearn.datasets import make_moons
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  roc_auc_score
from sklearn.datasets import make_moons
from sklearn.ensemble import GradientBoostingClassifier
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from odoo import models, fields, api, _
from odoo.exceptions import UserError
import joblib
import os
import logging
from datetime import datetime
import csv
import base64
from io import StringIO


# from odoo.addons.lead_scoring_ai.data.preprocessing_finalized import file_path

##########################################################

# Step 1: Load Data
def load_data(file_path):
    df = pd.read_csv(file_path)
    print(df.info())
    return df

# Step 2: Select Columns
def select_columns(df, columns):
    selected_df = df[columns]
    print(selected_df.head())
    return selected_df

# Step 3: Handle Missing Values
def fill_missing_country(df):
    # Extract unique City-Country mapping
    city_country_map = df.dropna(subset=['City', 'Country']).set_index('City')['Country'].to_dict()
    # Fill missing 'Country' values based on 'City'
    df['Country'] = df.apply(lambda row: city_country_map.get(row['City'], row['Country']), axis=1)
    print(city_country_map)
    return df

def handle_missing_specialization(df):
    # Fill missing values in 'Specialization'
    df['Specialization'] = df['Specialization'].fillna('Select')
    return df

# Step 4: Generate New Columns
def add_decision_authority(df):
    authority_levels = [5, 4, 3, 2, 1]
    probabilities = [0.05, 0.15, 0.25, 0.35, 0.20]
    df['Decision_Authority'] = np.random.choice(authority_levels, size=len(df), p=probabilities)
    return df

def add_job_position(df):
    job_titles_map = {
        5: ['CEO', 'CFO', 'CTO', 'COO', 'Chief Innovation Officer'],
        4: ['Vice President', 'Director of Operations', 'Head of Marketing', 'Senior Strategy Director'],
        3: ['Project Manager', 'Marketing Manager', 'Team Lead', 'Operations Manager'],
        2: ['Data Analyst', 'Software Engineer', 'Business Consultant', 'Technical Specialist'],
        1: ['Intern', 'Junior Developer', 'Assistant Coordinator', 'Associate Analyst']
    }
    df['Job Position'] = df['Decision_Authority'].apply(lambda x: np.random.choice(job_titles_map.get(x, ['Unknown'])))
    return df


def extract_decision_authority(df, job_titles_map):
    """
    Reads the 'Job Position' column and extracts the corresponding Decision Authority level
    based on the provided job_titles_map, creating a new column.

    Args:
    df (pd.DataFrame): The DataFrame containing the 'Job Position' column.
    job_titles_map (dict): A dictionary mapping Decision Authority levels to job titles.

    Returns:
    pd.DataFrame: The DataFrame with a new 'Extracted Decision Authority' column.
    """
    # Reverse the mapping: Job Title → Decision Authority
    reversed_map = {}
    for authority, titles in job_titles_map.items():
        for title in titles:
            reversed_map[title] = authority

    # Extract Decision Authority using the reversed mapping
    df['Extracted Decision Authority'] = df['Job Position'].map(reversed_map).fillna(0)  # Fill unknown titles with 0 or any default

    return df



# Step 5: Encode Binary Columns
def encode_binary_columns(df, columns, encoding_rules):
    for col in columns:
        encoded_col = f"{col} Encoded"
        df[encoded_col] = df[col].str.lower().map(encoding_rules).fillna(0)
    df.drop(columns, axis=1, inplace=True)
    return df

# Step 6: One-Hot Encoding
def one_hot_encode(df, columns):
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    encoded_data = encoder.fit_transform(df[columns])
    joblib.dump(encoder, 'fitted_encoder.joblib')

    # Set the index of the encoded DataFrame to match df's index
    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoder.get_feature_names_out(columns),
        index=df.index  # This is the key change
    )

    df = pd.concat([df.drop(columns, axis=1), encoded_df], axis=1)
    return df
##########################################################################################


import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from sklearn.preprocessing import OneHotEncoder


# Make sure that the following helper functions are defined/imported in your file:
# - fill_missing_country(df)
# - handle_missing_specialization(df)
# - add_job_position(df)
# - extract_decision_authority(df, job_titles_map)
# - encode_binary_columns(df, binary_columns, encoding_rules)
# - one_hot_encode(df, categorical_columns)

class JobPositionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, mode="train"):
        # mode can be "train" or "inference"
        self.mode = mode
        self.job_titles_map = {
            5: ['CEO', 'CFO', 'CTO', 'COO', 'Chief Innovation Officer'],
            4: ['Vice President', 'Director of Operations', 'Head of Marketing', 'Senior Strategy Director'],
            3: ['Project Manager', 'Marketing Manager', 'Team Lead', 'Operations Manager'],
            2: ['Data Analyst', 'Software Engineer', 'Business Consultant', 'Technical Specialist'],
            1: ['Intern', 'Junior Developer', 'Assistant Coordinator', 'Associate Analyst']
        }

    def fit(self, X, y=None):
        # No fitting needed in this transformer
        return self

    def transform(self, X):
        X = X.copy()
        # In training mode, generate the Job Position column
        if self.mode == "train":
            # Assumes add_job_position is defined externally and applies your business logic.
            X = add_decision_authority(X)
            X = add_job_position(X)
        # For both training and inference, extract decision authority
        X = extract_decision_authority(X, self.job_titles_map)
        # Drop the 'Job Position' column if it exists
        if 'Job Position' in X.columns:
            X.drop('Job Position', axis=1, inplace=True)
        return X


from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
import pandas as pd

class LeadPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self):
        # Define your rules and lists
        self.encoding_rules = {'yes': 1, 'y': 1, 'no': 0, 'n': 0}
        self.binary_columns = ['Through Recommendations', 'Do Not Call', 'Do Not Email']
        self.categorical_columns = ['Lead Source', 'Country', 'Tags', 'Specialization', 'City', 'Last Notable Activity']
        # List of identifier columns to drop.
        self.identifier_columns = ['id', 'Prospect ID', 'Lead Number']  # update as needed

        # Initialize the OneHotEncoder with handle_unknown='ignore'
        self.one_hot_enc = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')

    def fit(self, X, y=None):
        # Make a copy and drop unwanted identifier columns
        df = X.copy()
        df = df.drop([col for col in self.identifier_columns if col in df.columns], axis=1)

        # You could also apply your fill_missing_country and handle_missing_specialization here if needed:
        df = fill_missing_country(df)
        df = handle_missing_specialization(df)
        df = encode_binary_columns(df, self.binary_columns, self.encoding_rules)

        # Fit the one-hot encoder on categorical columns
        self.one_hot_enc.fit(df[self.categorical_columns])
        return self

    def transform(self, X):
        df = X.copy()
        # Drop identifier columns
        df = df.drop([col for col in self.identifier_columns if col in df.columns], axis=1)

        # Apply your preprocessing functions
        df = fill_missing_country(df)
        df = handle_missing_specialization(df)
        df = encode_binary_columns(df, self.binary_columns, self.encoding_rules)

        # For one-hot encoding, use the already fitted encoder
        encoded_data = self.one_hot_enc.transform(df[self.categorical_columns])
        encoded_df = pd.DataFrame(
            encoded_data,
            columns=self.one_hot_enc.get_feature_names_out(self.categorical_columns),
            index=df.index
        )
        # Drop original categorical columns and concatenate the encoded dataframe
        df = pd.concat([df.drop(self.categorical_columns, axis=1), encoded_df], axis=1)
        return df


from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

pipeline = Pipeline(steps=[
    ('job_position_processing', JobPositionTransformer(mode="train")),  # Keep your custom transformer here
    ('preprocessing', LeadPreprocessor()),
    ('classifier', RandomForestClassifier(n_estimators=500, random_state=42))
])

# Example usage:
# Assume that you have loaded your full dataset into a DataFrame `df`
# and that the target column in your dataset is called "Target".
file_path = r'C:\Program Files\Odoo 18.0.20250413\custom_addons\lead_scoring_ai\data\Lead Scoring.csv'
df = pd.read_csv(file_path)
# Split your data into features and target.

selected_columns= ['Lead Source','Do Not Email', 'Do Not Call', 'Country', 'City','Specialization', 'Through Recommendations' , 'Last Notable Activity','Tags']


X = df[selected_columns]
y = df['Converted']

# Further split into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Columns in X_train:", X_train.columns.tolist())



# # Train (fit) the pipeline using your training data.
pipeline.fit(X_train, y_train)
#
# # Evaluate the model on test data.
score = pipeline.score(X_test, y_test)
print("Test Score:", score)

# # Save the fully trained pipeline to disk for later inference.
joblib.dump(pipeline, 'trained_pipeline2.joblib')