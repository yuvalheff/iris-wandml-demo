import pandas as pd
import numpy as np
import pickle
import os

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from iris_wandml_project.config import ModelConfig


class ModelWrapper:
    def __init__(self, config: ModelConfig):
        self.config: ModelConfig = config
        self.model = None
        self.is_fitted = False
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the model based on config"""
        model_type = self.config.model_type.lower()
        model_params = self.config.model_params
        
        if model_type == 'svm':
            self.model = SVC(**model_params)
        elif model_type == 'logistic_regression':
            self.model = LogisticRegression(**model_params)
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(**model_params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def fit(self, X: pd.DataFrame, y: pd.Series):
        """
        Fit the classifier to the training data.

        Parameters:
        X: Training features.
        y: Target labels.

        Returns:
        self: Fitted classifier.
        """
        # Validate inputs
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        if not isinstance(y, pd.Series):
            raise ValueError("y must be a pandas Series")
        
        # Fit the model
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class labels for the input features.

        Parameters:
        X: Input features to predict.

        Returns:
        np.ndarray: Predicted class labels.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """
        Predict class probabilities for the input features.

        Parameters:
        X: Input features to predict probabilities.

        Returns:
        np.ndarray: Predicted class probabilities.
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        if not isinstance(X, pd.DataFrame):
            raise ValueError("X must be a pandas DataFrame")
        
        if not hasattr(self.model, 'predict_proba'):
            raise ValueError(f"Model {self.config.model_type} does not support predict_proba")
        
        return self.model.predict_proba(X)

    def get_model_info(self) -> dict:
        """
        Get information about the model.

        Returns:
        dict: Model information including type and parameters.
        """
        return {
            'model_type': self.config.model_type,
            'model_params': self.config.model_params,
            'is_fitted': self.is_fitted,
            'classes': getattr(self.model, 'classes_', None) if self.is_fitted else None
        }

    def save(self, path: str) -> None:
        """
        Save the model wrapper as an artifact

        Parameters:
        path (str): The file path to save the model.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, 'wb') as f:
            pickle.dump(self, f)

    def load(self, path: str) -> 'ModelWrapper':
        """
        Load the model wrapper from a saved artifact.

        Parameters:
        path (str): The file path to load the model from.

        Returns:
        ModelWrapper: The loaded model wrapper.
        """
        with open(path, 'rb') as f:
            return pickle.load(f)