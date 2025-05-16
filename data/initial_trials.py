#%% md
# Dataset Exploration
#%%
import pandas as pd

df = pd.read_csv('Lead Scoring.csv')
print(df.info())
#%% md
# Feature Selection step
#%%
selected_columns= ['Lead Source','Do Not Email', 'Do Not Call', 'Country', 'Country','Specialization', 'What is your current occupation', 'Through Recommendations' ,'City', 'I agree to pay the amount through cheque', 'Last Notable Activity','Converted']

selected_df= df[selected_columns]

print(selected_df.head())


#%% md
# Handling Missing Values
#%%
selected_df.isnull().sum()

#%% md
# A way to handle missing country and city values is to deduce them from each other, here The variable city_country_map now contains a dictionary of City → Country pairs, which allows quick lookup of which country a given city belongs to.
# 
# after running the below code, the missing values of countries dropped from 2461 to 838
# 
#%%

#%%
# Extract unique City-Country pairs from non-missing data
city_country_map = df.dropna(subset=['City', 'Country']).set_index('City')['Country'].to_dict()

print(city_country_map)
# Use the mapping to fill missing Country based on City
df['Country'] = df.apply(
    lambda row: city_country_map.get(row['City'], row['Country']),
    axis=1
)
#%%
print(city_country_map)
#%% md
# fll Missing Cities Using Country (Optional)
# For rows where Country is known but City is missing:Fills with most frequent city in a country
#%%
# Create a Country-City mapping (most frequent city per country)
country_city_map = df.dropna(subset=['City', 'Country']).groupby('Country')['City'].agg(pd.Series.mode).to_dict()
 # Ensure it has key-value pairs like {'Cairo': 'Egypt', 'Tokyo': 'Japan'}
# Fill missing City based on Country
df['City'] = df.apply(
    lambda row: country_city_map.get(row['Country'], row['City']),
    axis=1
)
#%%
selected_df.isnull().sum()

#%% md
# For Specialization, occupation, and lead source we will replace missing values with the mode:
#%%
df['Specialization'].value_counts(normalize=True) * 100 # check if there is a single dominant value

#%% md
# replacing missing specialization values with'select'
# 1. Preserves User Behavior
# "Select" directly reflects that the user did not choose a specialization from the dropdown/list.
# 
# This maintains the integrity of the data collection process (e.g., a blank or unselected field maps to "Select").
# 
# 2. Avoids Bias
# Using the mode (most frequent specialization) would artificially inflate the importance of that category.
# 
# Example: If "Marketing" is the mode, imputing it for 24% of missing values would mislead the model into thinking Marketing is more prevalent than it truly is.
# 
# 3. Retains Predictive Signal
# Leads who left Specialization as "Select" might behave differently (e.g., lower conversion rates).
# 
# The model can learn from this pattern if "Select" is treated as a distinct category.
#%%
df['Specialization'].fillna('Select', inplace=True)

#%%
df['Specialization'].value_counts(normalize=True) * 100 # notic that the percentage of the select value increased now

#%% md
# Let's do the same anlysis for the occupation column
#%%
df['What is your current occupation'].value_counts(normalize=True) * 100


#%% md
# Upon examination, we might omit the occupation column as it might lead to noise. the values in it are irrelevant to our B2B audience
# 
#%%
df.drop(columns=['What is your current occupation', 'Prospect ID', 'Lead Number', 'Lead Origin', 'TotalVisits', 'Total Time Spent on Website', 'Page Views Per Visit', 'Last Activity', 'How did you hear about X Education', 'What matters most to you in choosing a course', 'Search', 'Magazine', 'Newspaper Article', 'X Education Forums', 'Newspaper', 'Digital Advertisement', 'Receive More Updates About Our Courses', 'Lead Quality', 'Update me on Supply Chain Content', 'Get updates on DM Content', 'Lead Profile', 'Asymmetrique Activity Index', 'Asymmetrique Profile Index', 'Asymmetrique Activity Score', 'Asymmetrique Profile Score', 'A free copy of Mastering The Interview','I agree to pay the amount through cheque' ], inplace=True)
#%%
df.head()
#%%
df['Lead Source']
#%%
import numpy as np

# Define possible authority levels
authority_levels = [5, 4, 3, 2, 1]  # From highest to lowest

# Randomly assign decision authority
df['Decision_Authority'] = np.random.choice(authority_levels, size=len(df))

# Save the updated dataset
df.to_csv('updated_leads.csv', index=False)

print(df[['Decision_Authority']].head())

#%%
df['Decision_Authority'] = df['Lead Source'].map({
    'Referral': 5,  # Referrals might be direct senior-level contacts
    'LinkedIn': 4,  # LinkedIn tends to have mid-to-senior professionals
    'Website': 3,   # Website inquiries can be managers or specialists
    'Ad Campaign': 2,  # Cold leads often include analysts and entry-level roles
    'Other': 1      # Catch-all for low-authority leads
}).fillna(np.random.choice(authority_levels))  # Assign random if no match

#%%
df['Decision_Authority']
#%%
# Define job positions corresponding to authority levels
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
#%%
df
#%%
df['Decision_Authority'].value_counts()
#%% md
# The proposed method of using the lead Source turned out not to be a good approach because of the dominance of the Decision Authority value of 2. We researched another method, which is to randomize the values based on probability distribution of professional hierarchy.
#%%
import numpy as np

# Define authority levels and their probabilities (adjusted for balance)
authority_levels = [5, 4, 3, 2, 1]
probabilities = [0.05, 0.15, 0.25, 0.35, 0.20]  # More realistic distribution

# Assign Decision_Authority using weighted probability
df['Decision_Authority'] = np.random.choice(authority_levels, size=len(df), p=probabilities)

#%%
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
#%% md
# Insights: The probabilistic approach exhibits good variations that will help our model generalize.
#  Now it's time to finish the data preprocessing steps:
#%%
df
#%% md
# Handling Missing values
#%%
df.isnull().sum()
#%%
df['Tags']
#%%

#%% md
# Let's do One hot encoding for the categorical data columns. first we drop the job position column to reduce feature complexity. and then encode the rest
# 
#%%
df = df.drop('Job Position', axis=1)
#%%
df.head
#%%
df['Through Recommendations'].value_counts()
#%% md
# 
#%%
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

      # All others → 0
#%% md
# 
#%%
df.head(5)
#%%
# Drop irrelevant columns
cols_to_drop = ['Through Recommendations','Do Not Call', 'Do Not Email']
df = df.drop(cols_to_drop, axis=1)
#%%
df.head(5)
#%% md
# let's do some one hot encoding
#%%
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
#%%
df_encoded.head(5)
#%% md
# Let's Split the data
#%%
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

X = df_encoded.drop('Converted', axis=1)  # All features except target
y = df_encoded['Converted']

X.head(5)
y.head(5)

#%%

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,    # 20% for testing
    random_state=42,  # Ensures reproducibility
    stratify=y        # Preserves class balance in splits
)# Target variable
#%% md
# Training on Logistic Regression
#%%
model = LogisticRegression(max_iter=1000)  # Increase max_iter if needed
model.fit(X_train, y_train)
#%% md
# evaluation
#%%
# Predict on test set
y_pred = model.predict(X_test)

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
#%%
# 3. Get probabilities
y_prob = model.predict_proba(X_test)[:, 1]  # P(conversion)

# 4. Add probabilities back to DataFrame (for analysis)
test_df = X_test.copy()
test_df['conversion_prob'] = y_prob
test_df['actual'] = y_test  # Optional: Compare with ground truth

print(test_df[['conversion_prob', 'actual']].head(20))



#%% md
# Trying random forest
#%%
print(X_train.head(5))
y_train.head(5)
#%%
from sklearn.ensemble import RandomForestClassifier

# Initialize and train
rf = RandomForestClassifier(n_estimators=100, random_state=42)



rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

# Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
# Get raw probabilities (uncalibrated)
#%%
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




#%%
# Get predicted probabilities
proba = gb_model.predict_proba(X_test)

# Extract the probability of the positive class (e.g., lead becoming a sale)
lead_sale_prob = proba[:, 1]

# 4. Add probabilities back to DataFrame (for analysis)
test_df = X_test.copy()
test_df['conversion_prob'] = y_prob
test_df['actual'] = y_test  # Optional: Compare with ground truth

print(test_df[['conversion_prob', 'actual']].head(20))


#%% md
# Considering that all the models have similar outputs. Random forests seem to be an optimal choice since they capture non-linearity and are less computationally expensive than gradient boosting. Logistic regression yielded good scores but, in the future, if the dataset is more complex, it will probably fail to capture its patterns.
# 
# We will further fine-tune and calibrate it.
#%% md
# fine tuning
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
df.head()
#%%
df = df.rename(columns={'old_column_name': 'new_column_name'})

#%% md
# What we need to do is to rename the columns so they match with the fields names in Odoo itself.
# 