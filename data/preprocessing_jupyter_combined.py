#%%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import  roc_auc_score
from sklearn.datasets import make_moons
from sklearn.ensemble import GradientBoostingClassifier



#%%
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
    # Define job_titles_map
    job_titles_map = {
        5: ['CEO', 'CFO', 'CTO', 'COO', 'Chief Innovation Officer'],
        4: ['Vice President', 'Director of Operations', 'Head of Marketing', 'Senior Strategy Director'],
        3: ['Project Manager', 'Marketing Manager', 'Team Lead', 'Operations Manager'],
        2: ['Data Analyst', 'Software Engineer', 'Business Consultant', 'Technical Specialist'],
        1: ['Intern', 'Junior Developer', 'Assistant Coordinator', 'Associate Analyst']
    }
    selected_columns = [
        'Lead Source', 'Do Not Email', 'Do Not Call', 'Country', 'City','Specialization',
         'Through Recommendations', 'Job Position'
         , 'Last Notable Activity','Tags', 'Converted'
    ]


    # Load and preprocess data
    df = load_data(file_path)
    df = add_decision_authority(df)
    df = add_job_position(df)
    df = select_columns(df, selected_columns)
    df = fill_missing_country(df)
    df = handle_missing_specialization(df)
    df = extract_decision_authority(df,job_titles_map)
    df.drop(['Job Position'], axis=1, inplace=True)


    print(df.head(5))

    encoding_rules = {'yes': 1, 'y': 1, 'no': 0, 'n': 0}
    binary_columns = ['Through Recommendations', 'Do Not Call', 'Do Not Email']
    df = encode_binary_columns(df, binary_columns, encoding_rules)

    print(df.head(5))

    categorical_columns = ['Lead Source', 'Country', 'Tags', 'Specialization', 'City', 'Last Notable Activity']
    df = one_hot_encode(df, categorical_columns)

    # Save processed data
    df.to_csv('processed_leads.csv', index=False)

file_path = r'C:\Users\islam\Odoo Intern\CRM Project\CRM-Sales-Prediction\archive\Lead Scoring.csv'
main_pipeline(file_path)



#%%

#%% md
# Splitting Data
#%%

df = load_data(r'C:\Users\islam\Odoo Intern\CRM Project\CRM-Sales-Prediction\archive\processed_leads.csv')

y = df['Converted']
X = df.drop('Converted', axis=1)  # All features except target


# Use same random_state and stratify
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,  # Must match original
    stratify=y       # Must match original
)
#%%
print(X_train.head(5))
y_train.head(5)
#%% md
# Logistic Regression
#%%
model = LogisticRegression(max_iter=1000)  # Increase max_iter if needed
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

print(classification_report(y_test, y_pred))
# 3. Get probabilities
y_prob = model.predict_proba(X_test)[:, 1]  # P(conversion)

# 4. Add probabilities back to DataFrame (for analysis)
test_df = X_test.copy()
test_df['conversion_prob'] = y_prob
test_df['actual'] = y_test  # Optional: Compare with ground truth

print(test_df[['conversion_prob', 'actual']].head(20))


#%% md
# RandomForests
#%%
# Initialize and train
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
# Get raw probabilities (uncalibrated)

# Get raw probabilities (uncalibrated)
y_prob_uncalibrated = rf.predict_proba(X_test)[:, 1]

print(y_prob_uncalibrated)

# 4. Add probabilities back to DataFrame (for analysis)
test_df = X_test.copy()
test_df['conversion_prob'] = y_prob_uncalibrated
test_df['actual'] = y_test  # Optional: Compare with ground truth

print(test_df[['conversion_prob', 'actual']].head(20))
#%% md
# Trying out both models on synthetic non-linear dataset, RF outperformed LR as we see.
#%%

from sklearn.metrics import  roc_auc_score
from sklearn.datasets import make_moons
X_nl, y_nl = make_moons(n_samples=1000, noise=0.3, random_state=42)

# Train both models
lr_nl = LogisticRegression().fit(X_nl, y_nl)
rf_nl = RandomForestClassifier(random_state=42).fit(X_nl, y_nl)

# Compare AUC
print("LR AUC (Non-linear):", roc_auc_score(y_nl, lr_nl.predict_proba(X_nl)[:, 1]))  # Likely poor
print("RF AUC (Non-linear):", roc_auc_score(y_nl, rf_nl.predict_proba(X_nl)[:, 1]))  # Should be >0.9
#%% md
# GradientBoosting
#%%
from sklearn.ensemble import GradientBoostingClassifier


# Initialize the Gradient Boosting Regressor
gb_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42)

# Fit the model to your training data
gb_model.fit(X_train, y_train)

# Make predictions
y_pred = gb_model.predict(X_test)

# Evaluate the model
print(f"Accuracy: {accuracy_score(y_test, y_pred):.3f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# Get predicted probabilities
proba = gb_model.predict_proba(X_test)

# Extract the probability of the positive class (e.g., lead becoming a sale)
lead_sale_prob = proba[:, 1]

# 4. Add probabilities back to DataFrame (for analysis)
test_df = X_test.copy()
test_df['conversion_prob'] = lead_sale_prob
test_df['actual'] = y_test  # Optional: Compare with ground truth

print(test_df[['conversion_prob', 'actual']].head(20))




#%% md
# Considering that all the models have similar outputs. Random forests seem to be an optimal choice since they capture non-linearity and are less computationally expensive than gradient boosting. Logistic regression yielded good scores but, in the future, if the dataset is more complex, it will probably fail to capture its patterns.
# 
# We will further fine-tune and calibrate it.
#%% md
# 
#%%
# # first accuracy:  0.9128
# first f1: 0.88


#  2nd accuracy: 0.9134 , n_estimators=200
#  2nd f1: 0.89

# 3rd accuracy:0.9139  , n_estimators=500
# 3rd f1: 0.89

# according to available litrature, the obtained f1 score is considered good.

from sklearn.ensemble import RandomForestClassifier

# Initialize and train
rf = RandomForestClassifier(n_estimators=500, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
# Get raw probabilities (uncalibrated)
#%%


#%% md
# Calibrating Random Forests
# 
# Problem: RFs tend to produce overconfident probabilities (e.g., predicting 0.95 when true likelihood is 0.7)
# 
# Solution: Use Platt Scaling (logistic calibration) or Isotonic Regression to align probabilities with actual outcomes
#%%

#%%

#%%
from sklearn.calibration import CalibratedClassifierCV

# Calibrate with Platt Scaling (sigmoid)
calibrated_rf = CalibratedClassifierCV(
    estimator=rf,
    method='sigmoid',  # or 'isotonic' for non-parametric
    cv=5,             # Use 5-fold cross-validation
    ensemble=True
)

calibrated_rf.fit(X_train, y_train)


#%%
# Get raw probabilities (uncalibrated)
y_prob_uncalibrated = rf.predict_proba(X_test)[:, 1]

print(y_prob_uncalibrated)

# 4. Add probabilities back to DataFrame (for analysis)
test_df = X_test.copy()
test_df['uncalibrated_prob'] = y_prob_uncalibrated
test_df['actual'] = y_test  # Optional: Compare with ground truth

#print(test_df[['uncalibrated_prob', 'actual']].head(20))


# Get predicted probabilities
probab = calibrated_rf.predict_proba(X_test)

# Extract the probability of the positive class (e.g., lead becoming a sale)
y_prob_calibrated = probab[:, 1]

test_df['rf_calibrated_prob'] = y_prob_calibrated

print(test_df[['rf_calibrated_prob', 'uncalibrated_prob', 'actual']].head(20))


#%%
import joblib

joblib.dump(calibrated_rf, 'calibrated_rf_odoo.joblib')
#%%
