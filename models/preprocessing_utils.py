# custom_addons/lead_scoring_ai/models/preprocessing_utils.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import joblib

def load_data(file_path):
    df = pd.read_csv(file_path)
    print(df.info())
    return df

def select_columns(df, columns):
    selected_df = df[columns]
    print(selected_df.head())
    return selected_df

def fill_missing_country(df):
    city_country_map = df.dropna(subset=['City', 'Country']).set_index('City')['Country'].to_dict()
    df['Country'] = df.apply(lambda row: city_country_map.get(row['City'], row['Country']), axis=1)
    print(city_country_map)
    return df

def handle_missing_specialization(df):
    df['Specialization'] = df['Specialization'].fillna('Select')
    return df

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
    reversed_map = {}
    for authority, titles in job_titles_map.items():
        for title in titles:
            reversed_map[title] = authority
    df['Extracted Decision Authority'] = df['Job Position'].map(reversed_map).fillna(0)
    return df

def encode_binary_columns(df, columns, encoding_rules):
    for col in columns:
        encoded_col = f"{col} Encoded"
        df[encoded_col] = df[col].str.lower().map(encoding_rules).fillna(0)
    df.drop(columns, axis=1, inplace=True)
    return df

def one_hot_encode(df, columns):
    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    encoded_data = encoder.fit_transform(df[columns])
    joblib.dump(encoder, 'fitted_encoder.joblib')
    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoder.get_feature_names_out(columns),
        index=df.index
    )
    df = pd.concat([df.drop(columns, axis=1), encoded_df], axis=1)
    return df
