import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

# === Custom Transformer: Drop High-Cardinality Columns ===
class HighCardinalityDropper(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=100):
        self.threshold = threshold
        self.columns_to_drop_ = []

    def fit(self, X, y=None):
        cat_cols = X.select_dtypes(include=['object', 'category']).columns
        self.columns_to_drop_ = [col for col in cat_cols if X[col].nunique() > self.threshold]
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop_, errors='ignore')


# === Custom Transformer: Drop Leakage or ID Columns ===
class ColumnDropper(BaseEstimator, TransformerMixin):
    def __init__(self, cols_to_drop):
        self.cols_to_drop = cols_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.cols_to_drop, errors='ignore')


# === Function: Build the Preprocessing Pipeline ===
def build_preprocessing_pipeline(X_train):
    # Drop high-cardinality first to identify encodable columns
    temp_dropper = HighCardinalityDropper(threshold=100)
    X_reduced = temp_dropper.fit_transform(X_train.copy())

    cat_cols = X_reduced.select_dtypes(include=['object', 'category']).columns
    low_card_cols = [col for col in cat_cols if X_reduced[col].nunique() <= 10]
    high_card_cols = [col for col in cat_cols if X_reduced[col].nunique() > 10]

    leakage_cols = [
        'state_longitude',
        'user_longitude',
        'delivery_time_improvement',
        'estimated_new_delivery_time',
        'distance_dc_to_user_km',
        'distance_new_dc_to_user_km'
    ]
    id_cols = ['order_id', 'product_id', 'distribution_center_id']
    drop_cols = leakage_cols + id_cols

    column_transformer = ColumnTransformer(transformers=[
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'), low_card_cols),
        ('label', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), high_card_cols)
    ], remainder='passthrough')

    pipeline = Pipeline(steps=[
        ('drop_high_card_cols', HighCardinalityDropper(threshold=100)),
        ('drop_leakage', ColumnDropper(cols_to_drop=drop_cols)),
        ('encode_columns', column_transformer)
    ])

    return pipeline
