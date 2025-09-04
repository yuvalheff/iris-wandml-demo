"""
ML Pipeline for Iris Classification

Complete pipeline that combines data processing, feature engineering, and modeling
for deployment as an MLflow model.
"""

import pandas as pd
import numpy as np
from typing import Union, List

from iris_wandml_project.pipeline.data_preprocessing import DataProcessor
from iris_wandml_project.pipeline.feature_preprocessing import FeatureProcessor
from iris_wandml_project.pipeline.model import ModelWrapper


class ModelPipeline:
    """
    Complete ML pipeline for iris classification that combines all processing steps.
    
    This class handles the complete flow from raw data input to final predictions,
    making it suitable for MLflow model registry deployment.
    """
    
    def __init__(self, data_processor: DataProcessor = None, 
                 feature_processor: FeatureProcessor = None,
                 model_wrapper: ModelWrapper = None):
        """
        Initialize the pipeline with fitted components.
        
        Parameters:
        data_processor: Fitted DataProcessor instance
        feature_processor: Fitted FeatureProcessor instance  
        model_wrapper: Fitted ModelWrapper instance
        """
        self.data_processor = data_processor
        self.feature_processor = feature_processor
        self.model_wrapper = model_wrapper
        
        # Validation
        if data_processor and not data_processor.is_fitted:
            raise ValueError("DataProcessor must be fitted")
        if feature_processor and not feature_processor.is_fitted:
            raise ValueError("FeatureProcessor must be fitted") 
        if model_wrapper and not model_wrapper.is_fitted:
            raise ValueError("ModelWrapper must be fitted")
    
    def predict(self, X: Union[pd.DataFrame, dict, List[dict]]) -> np.ndarray:
        """
        Make predictions on input data.
        
        Parameters:
        X: Input data - can be DataFrame, single dict, or list of dicts
        
        Returns:
        np.ndarray: Predicted class labels (0, 1, 2)
        """
        # Convert input to DataFrame if needed
        X_df = self._prepare_input(X)
        
        # Apply data processing (extract features, no target needed for prediction)
        X_features = self._apply_data_processing(X_df)
        
        # Apply feature processing (scaling)
        X_processed = self.feature_processor.transform(X_features)
        
        # Make predictions
        predictions = self.model_wrapper.predict(X_processed)
        
        return predictions
    
    def predict_proba(self, X: Union[pd.DataFrame, dict, List[dict]]) -> np.ndarray:
        """
        Predict class probabilities for input data.
        
        Parameters:
        X: Input data - can be DataFrame, single dict, or list of dicts
        
        Returns:
        np.ndarray: Predicted class probabilities with shape (n_samples, n_classes)
        """
        # Convert input to DataFrame if needed
        X_df = self._prepare_input(X)
        
        # Apply data processing (extract features, no target needed for prediction)
        X_features = self._apply_data_processing(X_df)
        
        # Apply feature processing (scaling)
        X_processed = self.feature_processor.transform(X_features)
        
        # Make probability predictions
        probabilities = self.model_wrapper.predict_proba(X_processed)
        
        return probabilities
    
    def predict_with_labels(self, X: Union[pd.DataFrame, dict, List[dict]]) -> List[str]:
        """
        Make predictions and return original species labels.
        
        Parameters:
        X: Input data - can be DataFrame, single dict, or list of dicts
        
        Returns:
        List[str]: Predicted species names
        """
        # Get numeric predictions
        predictions = self.predict(X)
        
        # Convert back to original labels
        predictions_series = pd.Series(predictions)
        original_labels = self.data_processor.inverse_transform_target(predictions_series)
        
        return original_labels.tolist()
    
    def _prepare_input(self, X: Union[pd.DataFrame, dict, List[dict]]) -> pd.DataFrame:
        """
        Convert various input formats to a standardized DataFrame.
        
        Parameters:
        X: Input data in various formats
        
        Returns:
        pd.DataFrame: Standardized DataFrame
        """
        if isinstance(X, pd.DataFrame):
            return X
        elif isinstance(X, dict):
            return pd.DataFrame([X])
        elif isinstance(X, list):
            return pd.DataFrame(X)
        else:
            raise ValueError(f"Unsupported input type: {type(X)}")
    
    def _apply_data_processing(self, X_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply data processing for prediction (features only, no target).
        
        Parameters:
        X_df: Input DataFrame
        
        Returns:
        pd.DataFrame: Processed features
        """
        # For prediction, we only need feature extraction
        # Check if the input has the required feature columns
        feature_columns = self.data_processor.feature_columns
        
        if all(col in X_df.columns for col in feature_columns):
            # Input has all feature columns, extract them directly
            return X_df[feature_columns].copy()
        else:
            raise ValueError(f"Input must contain all required feature columns: {feature_columns}")
    
    def get_feature_names(self) -> List[str]:
        """
        Get the names of input features expected by the pipeline.
        
        Returns:
        List[str]: List of feature column names
        """
        return self.data_processor.feature_columns
    
    def get_target_classes(self) -> List[str]:
        """
        Get the possible target class names.
        
        Returns:
        List[str]: List of target class names
        """
        return self.data_processor.get_target_classes()
    
    def get_pipeline_info(self) -> dict:
        """
        Get information about the pipeline components.
        
        Returns:
        dict: Pipeline component information
        """
        return {
            'data_processor': {
                'feature_columns': self.data_processor.feature_columns,
                'target_column': self.data_processor.target_column,
                'target_classes': self.get_target_classes()
            },
            'feature_processor': {
                'apply_scaling': self.feature_processor.config.apply_scaling,
                'scaling_method': self.feature_processor.config.scaling_method
            },
            'model_wrapper': self.model_wrapper.get_model_info()
        }
