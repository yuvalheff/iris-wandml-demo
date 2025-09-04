from typing import Optional, Tuple
import pandas as pd
import pickle
import os

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder

from iris_wandml_project.config import DataConfig


class DataProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, config: DataConfig):
        self.config: DataConfig = config
        self.label_encoder = None
        self.feature_columns = None
        self.target_column = None
        self.is_fitted = False

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> 'DataProcessor':
        """
        Fit the data processor to the data.

        Parameters:
        X (pd.DataFrame): The input data with features and target.
        y (Optional[pd.Series]): Not used (target is in X).

        Returns:
        DataProcessor: The fitted processor.
        """
        # Validate input
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame")
        
        # Store feature and target columns based on config
        self.feature_columns = self.config.feature_columns
        self.target_column = self.config.target_column
        
        # Validate required columns exist
        required_columns = self.feature_columns + [self.target_column]
        missing_columns = set(required_columns) - set(X.columns)
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        # Fit label encoder on target column
        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(X[self.target_column])
        
        self.is_fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Transform the input data based on the configuration.

        Parameters:
        X (pd.DataFrame): The input data to transform.

        Returns:
        Tuple[pd.DataFrame, pd.Series]: The transformed features and encoded target.
        """
        if not self.is_fitted:
            raise ValueError("DataProcessor must be fitted before transform")
        
        # Validate input
        if not isinstance(X, pd.DataFrame):
            raise ValueError("Input X must be a pandas DataFrame")
        
        # Extract features (remove Id column and select feature columns)
        X_features = X[self.feature_columns].copy()
        
        # Extract and encode target
        y_encoded = self.label_encoder.transform(X[self.target_column])
        y_encoded = pd.Series(y_encoded, index=X.index, name=self.target_column)
        
        return X_features, y_encoded

    def fit_transform(self, X: pd.DataFrame, y: Optional[pd.Series] = None, **fit_params) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Fit and transform the input data.

        Parameters:
        X (pd.DataFrame): The input data.
        y (Optional[pd.Series]): Not used.

        Returns:
        Tuple[pd.DataFrame, pd.Series]: The transformed features and encoded target.
        """
        return self.fit(X, y).transform(X)

    def inverse_transform_target(self, y_encoded: pd.Series) -> pd.Series:
        """
        Convert encoded target back to original labels.

        Parameters:
        y_encoded (pd.Series): The encoded target values.

        Returns:
        pd.Series: The original target labels.
        """
        if not self.is_fitted:
            raise ValueError("DataProcessor must be fitted before inverse_transform_target")
        
        return pd.Series(self.label_encoder.inverse_transform(y_encoded), 
                        index=y_encoded.index, name=self.target_column)

    def get_target_classes(self) -> list:
        """
        Get the original target class names.

        Returns:
        list: The original target class names.
        """
        if not self.is_fitted:
            raise ValueError("DataProcessor must be fitted before getting target classes")
        
        return list(self.label_encoder.classes_)

    def save(self, path: str) -> None:
        """
        Save the data processor as an artifact

        Parameters:
        path (str): The file path to save the processor.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path: str) -> 'DataProcessor':
        """
        Load the data processor from a saved artifact.

        Parameters:
        path (str): The file path to load the processor from.

        Returns:
        DataProcessor: The loaded data processor.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)
