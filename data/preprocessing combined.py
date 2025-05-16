import pandas as pd

df = pd.read_csv('Lead Scoring.csv')
print(df.info())

selected_columns= ['Lead Source','Do Not Email', 'Do Not Call', 'Country', 'Country','Specialization', 'What is your current occupation', 'Through Recommendations' ,'City', 'I agree to pay the amount through cheque', 'Last Notable Activity','Converted']

selected_df= df[selected_columns]

print(selected_df.head())

# checking for nyll values

selected_df.isnull().sum()

# #%% md
# A way to handle missing country and city values is to deduce them from each other, here The variable city_country_map now contains a dictionary of City → Country pairs, which allows quick lookup of which country a given city belongs to.
#
# after running the below code, the missing values of countries dropped from 2461 to 838
#
# Extract unique City-Country pairs from non-missing data
city_country_map = df.dropna(subset=['City', 'Country']).set_index('City')['Country'].to_dict()

print(city_country_map)
# Use the mapping to fill missing Country based on City
df['Country'] = df.apply(
    lambda row: city_country_map.get(row['City'], row['Country']),
    axis=1
)


print(city_country_map)



df['Specialization'].value_counts(normalize=True) * 100 # check if there is a single dominant value

df['Specialization'].fillna('Select', inplace=True)

import numpy as np

import numpy as np

# Define authority levels and their probabilities (adjusted for balance)
authority_levels = [5, 4, 3, 2, 1]
probabilities = [0.05, 0.15, 0.25, 0.35, 0.20]  # More realistic distribution

# Assign Decision_Authority using weighted probability
df['Decision_Authority'] = np.random.choice(authority_levels, size=len(df), p=probabilities)


job_titles_map = {
    5: ['CEO', 'CFO', 'CTO', 'COO', 'Chief Innovation Officer'],
    4: ['Vice President', 'Director of Operations', 'Head of Marketing', 'Senior Strategy Director'],
    3: ['Project Manager', 'Marketing Manager', 'Team Lead', 'Operations Manager'],
    2: ['Data Analyst', 'Software Engineer', 'Business Consultant', 'Technical Specialist'],
    1: ['Intern', 'Junior Developer', 'Assistant Coordinator', 'Associate Analyst']
}


# Generate 'Job Position' column by assigning random titles from the dictionary
df['Job Position'] = df['Decision_Authority'].apply(lambda x: np.random.choice(job_titles_map.get(x, ['Unknown'])))

# Save the updated dataset
df.to_csv('updated_leads.csv', index=False)

print(df[['Decision_Authority', 'Job Position']].head())




# encoding binary columns of yes/no
# columns to be encoded:
# 'Through Recommendations'
# 'Do Not Call'
# 'Do Not Email'


encoding_rules = {
    'yes': 1,
    'y': 1,
    'no': 0,
    'n': 0,
}

df['Through Recommendations Encoded'] = df['Through Recommendations'].str.lower().map (encoding_rules).fillna(0)
df['Do Not Call Encoded'] = df['Do Not Call'].str.lower().map (encoding_rules).fillna(0)
df['Do Not Email Encoded'] = df['Do Not Email'].str.lower().map (encoding_rules).fillna(0)


# Drop irrelevant columns
cols_to_drop = ['Through Recommendations','Do Not Call', 'Do Not Email']
df = df.drop(cols_to_drop, axis=1)


from sklearn.preprocessing import OneHotEncoder

# Initialize encoder
encoder = OneHotEncoder(sparse_output=False, drop='first')  # Drops first category to avoid multicollinearity

# Fit and transform
encoded_data = encoder.fit_transform(df[['Lead Source', 'Country', 'Specialization', 'Tags', 'City', 'Last Notable Activity']])

# Convert to DataFrame
encoded_df = pd.DataFrame(
    encoded_data,
    columns=encoder.get_feature_names_out(['Lead Source', 'Country', 'Specialization', 'Tags', 'City', 'Last Notable Activity'])
)

# Combine with original data
df_encoded = pd.concat([df.drop(['Lead Source', 'Country', 'Specialization', 'Tags', 'City', 'Last Notable Activity'], axis=1), encoded_df], axis=1)

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

X = df_encoded.drop('Converted', axis=1)  # All features except target
y = df_encoded['Converted']

X.head(5)
y.head(5)


      # All others → 0





