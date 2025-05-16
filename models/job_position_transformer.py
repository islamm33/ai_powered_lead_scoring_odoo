# custom_addons/lead_scoring_ai/models/job_position_transformer.py
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from odoo.addons.lead_scoring_ai.models.preprocessing_utils import add_decision_authority, add_job_position, extract_decision_authority

class JobPositionTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, mode="train"):
        self.mode = mode
        self.job_titles_map = {
            5: ['CEO', 'CFO', 'CTO', 'COO', 'Chief Innovation Officer'],
            4: ['Vice President', 'Director of Operations', 'Head of Marketing', 'Senior Strategy Director'],
            3: ['Project Manager', 'Marketing Manager', 'Team Lead', 'Operations Manager'],
            2: ['Data Analyst', 'Software Engineer', 'Business Consultant', 'Technical Specialist'],
            1: ['Intern', 'Junior Developer', 'Assistant Coordinator', 'Associate Analyst']
        }

    def fit(self, X, y=None):
        return self



    def transform(self, X):
        X = X.copy()
        if self.mode == "train":
            X = add_decision_authority(X)
            X = add_job_position(X)
        X = extract_decision_authority(X, self.job_titles_map)
        if 'Job Position' in X.columns:
            X.drop('Job Position', axis=1, inplace=True)
        return X
