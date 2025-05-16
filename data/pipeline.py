import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder


# Step 1: Load Data
def load_data(file_path):
    df = pd.read_csv(file_path)
    print(df.info())
    return df


# Step 2: Select Columns
def select_columns(df, columns):
    selected_df = df[columns].copy()
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
    df['Extracted Decision Authority'] = df['Job Position'].map(reversed_map).fillna(
        0)  # Fill unknown titles with 0 or any default

    return df



# Define job_titles_map
job_titles_map = {
    5: ['CEO', 'CFO', 'CTO', 'COO', 'Chief Innovation Officer'],
    4: ['Vice President', 'Director of Operations', 'Head of Marketing', 'Senior Strategy Director'],
    3: ['Project Manager', 'Marketing Manager', 'Team Lead', 'Operations Manager'],
    2: ['Data Analyst', 'Software Engineer', 'Business Consultant', 'Technical Specialist'],
    1: ['Intern', 'Junior Developer', 'Assistant Coordinator', 'Associate Analyst']
}



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
    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoder.get_feature_names_out(columns)
    )
    df = pd.concat([df.drop(columns, axis=1), encoded_df], axis=1)
    return df


# Step 7: Split Data
def split_data(df, target_column):
    X = df.drop(target_column, axis=1)
    y = df[target_column]
    return X, y


# Main Pipeline
def main_pipeline(file_path):
    # Load and preprocess data
    df = load_data(file_path)
    selected_columns = [
        'Lead Source', 'Do Not Email', 'Do Not Call', 'Country', 'Specialization',
        'What is your current occupation', 'Through Recommendations', 'City',
        'I agree to pay the amount through cheque', 'Last Notable Activity', 'Converted'
    ]
    df = select_columns(df, selected_columns)

    df.head()
    df.isnull().sum()


    df = fill_missing_country(df)
    df.isnull().sum()

    df = handle_missing_specialization(df)
    df.head()
    df = add_decision_authority(df)
    df.head()
    df = add_job_position(df)
    df.head()
    encoding_rules = {'yes': 1, 'y': 1, 'no': 0, 'n': 0}
    binary_columns = ['Through Recommendations', 'Do Not Call', 'Do Not Email']
    df = encode_binary_columns(df, binary_columns, encoding_rules)

    categorical_columns = ['Lead Source', 'Country', 'Specialization', 'City', 'Last Notable Activity']
    df = one_hot_encode(df, categorical_columns)

    # Save processed data
    df.to_csv('processed_leads.csv', index=False)

    df.head (10)

main_pipeline(r'C:\Users\islam\Odoo Intern\CRM Project\CRM-Sales-Prediction\archive\Lead Scoring.csv')

