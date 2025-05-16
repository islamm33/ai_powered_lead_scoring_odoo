# custom_addons/lead_scoring_ai/models/lead_preprocessor.py
import os
import joblib
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import OneHotEncoder
from .preprocessing_utils import fill_missing_country, handle_missing_specialization, encode_binary_columns


class LeadPreprocessor(BaseEstimator, TransformerMixin):
    def __init__(self, encoder_path='./fitted_encoder.joblib'):
        self.encoding_rules = {'yes': 1, 'y': 1, 'no': 0, 'n': 0}
        self.binary_columns = ['Through Recommendations', 'Do Not Call', 'Do Not Email']
        self.categorical_columns = ['Lead Source', 'Country', 'Tags', 'Specialization', 'City', 'Last Notable Activity']

        self.identifier_columns = ['id', 'Prospect ID', 'Lead Number']
        self.encoder_path = encoder_path

        # Initialize encoder with production-safe parameters
        self.one_hot_enc = OneHotEncoder(
            sparse_output=False,
            drop='first',
            handle_unknown='ignore',  # Critical for inference stability
            categories='auto'
        )

    def fit(self, X, y=None):
        # Training mode: full preprocessing and encoder fitting
        df = self._base_preprocessing(X, training_mode=True)

        # Fit encoder and persist state
        self.one_hot_enc.fit(df[self.categorical_columns])
        self._persist_encoder()

        return self

    def transform(self, X):
        # Inference mode: load encoder if not already loaded
        if not hasattr(self.one_hot_enc, 'categories_'):
            self._load_encoder()

        df = self._base_preprocessing(X)
        df = self._safe_encode_categoricals(df)
        return df

    def _base_preprocessing(self, X, training_mode=False):
        """Common preprocessing steps for both fit and transform"""
        df = X.copy()

        # 1. Remove identifiers
        df = df.drop([col for col in self.identifier_columns if col in df.columns], axis=1)

        # 2. Handle missing values
        df = fill_missing_country(df)
        df = handle_missing_specialization(df)

        # 3. Binary encoding
        df = encode_binary_columns(df, self.binary_columns, self.encoding_rules)

        # 4. During training: validate categorical columns
        if training_mode:
            self._validate_categorical_columns(df)

        return df

    def _safe_encode_categoricals(self, df):
        """Robust encoding with production safeguards"""
        # Handle unseen categories
        for i, col in enumerate(self.categorical_columns):
            valid_cats = self.one_hot_enc.categories_[i]
            df[col] = df[col].apply(lambda x: x if x in valid_cats else 'UNKNOWN')

        # Transform using persisted encoder
        encoded_data = self.one_hot_enc.transform(df[self.categorical_columns])
        encoded_df = pd.DataFrame(
            encoded_data,
            columns=self.one_hot_enc.get_feature_names_out(self.categorical_columns),
            index=df.index
        )

        return pd.concat([df.drop(self.categorical_columns, axis=1), encoded_df], axis=1)

    def _persist_encoder(self):
        """Save encoder state with metadata"""
        joblib.dump({
            'encoder': self.one_hot_enc,
            'categorical_columns': self.categorical_columns,
            'binary_columns': self.binary_columns
        }, self.encoder_path)

    def _load_encoder(self):
        """Load encoder state for inference"""
        if not os.path.exists(self.encoder_path):
            raise FileNotFoundError(f"Encoder not found at {self.encoder_path}. Train first!")

        enc_data = joblib.load(self.encoder_path)
        self.one_hot_enc = enc_data['encoder']
        self.categorical_columns = enc_data['categorical_columns']
        self.binary_columns = enc_data['binary_columns']

    def _validate_categorical_columns(self, df):
        """Ensure no missing categorical columns during training"""
        missing = set(self.categorical_columns) - set(df.columns)
        if missing:
            raise ValueError(f"Missing categorical columns in training data: {missing}")