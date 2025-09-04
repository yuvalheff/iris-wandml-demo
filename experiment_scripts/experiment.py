from pathlib import Path
import pandas as pd
import numpy as np
import os
import pickle
import json
import sklearn
import mlflow

from iris_wandml_project.pipeline.feature_preprocessing import FeatureProcessor
from iris_wandml_project.pipeline.data_preprocessing import DataProcessor
from iris_wandml_project.pipeline.model import ModelWrapper
from iris_wandml_project.config import Config
from experiment_scripts.evaluation import ModelEvaluator
from iris_wandml_project.model_pipeline import ModelPipeline

DEFAULT_CONFIG = str(Path(__file__).parent / 'config.yaml')


class Experiment:
    def __init__(self):
        self._config = Config.from_yaml(DEFAULT_CONFIG)
        
    def run(self, train_dataset_path, test_dataset_path, output_dir, seed=42):
        """
        Run the complete experiment pipeline.
        
        Parameters:
        train_dataset_path: Path to training data
        test_dataset_path: Path to test data  
        output_dir: Directory to save outputs
        seed: Random seed for reproducibility
        
        Returns:
        dict: Experiment results in the required format
        """
        print(f"🚀 Starting Iris Classification Experiment")
        print(f"📊 Config: {self._config}")
        print(f"🌱 Random seed: {seed}")
        
        # Set random seed for reproducibility
        np.random.seed(seed)
        
        # Create output directories
        plots_dir = os.path.join(output_dir, "output", "plots")
        model_artifacts_dir = os.path.join(output_dir, "output", "model_artifacts")
        general_artifacts_dir = os.path.join(output_dir, "output", "general_artifacts")
        
        os.makedirs(plots_dir, exist_ok=True)
        os.makedirs(model_artifacts_dir, exist_ok=True)
        os.makedirs(general_artifacts_dir, exist_ok=True)
        
        try:
            # Load data
            print("📂 Loading datasets...")
            train_data = pd.read_csv(train_dataset_path)
            test_data = pd.read_csv(test_dataset_path)
            print(f"   Training data shape: {train_data.shape}")
            print(f"   Test data shape: {test_data.shape}")
            
            # Initialize components
            print("🔧 Initializing pipeline components...")
            data_processor = DataProcessor(self._config.data_prep)
            feature_processor = FeatureProcessor(self._config.feature_prep)
            model_wrapper = ModelWrapper(self._config.model)
            evaluator = ModelEvaluator(self._config.model_evaluation)
            
            # Data processing
            print("🔄 Processing training data...")
            X_train, y_train = data_processor.fit_transform(train_data)
            print(f"   Processed training features shape: {X_train.shape}")
            print(f"   Target classes: {data_processor.get_target_classes()}")
            
            # Feature processing
            print("⚙️ Processing features...")
            X_train_processed = feature_processor.fit_transform(X_train)
            print(f"   Feature processing complete. Shape: {X_train_processed.shape}")
            
            # Model training
            print("🎯 Training model...")
            model_wrapper.fit(X_train_processed, y_train)
            print(f"   Model training complete: {model_wrapper.get_model_info()['model_type']}")
            
            # Cross-validation
            print("📊 Performing cross-validation...")
            cv_results = evaluator.cross_validate_model(model_wrapper.model, X_train_processed, y_train)
            print(f"   CV Accuracy: {cv_results['cv_accuracy_mean']:.4f} ± {cv_results['cv_accuracy_std']:.4f}")
            print(f"   CV AUC: {cv_results['cv_auc_mean']:.4f} ± {cv_results['cv_auc_std']:.4f}")
            
            # Test evaluation
            print("🧪 Evaluating on test data...")
            X_test, y_test = data_processor.transform(test_data)
            X_test_processed = feature_processor.transform(X_test)
            
            # Make predictions
            y_pred = model_wrapper.predict(X_test_processed)
            y_proba = model_wrapper.predict_proba(X_test_processed)
            
            # Calculate metrics
            test_metrics = evaluator.calculate_metrics(
                y_test, y_pred, y_proba, data_processor.get_target_classes()
            )
            
            print(f"   Test Accuracy: {test_metrics['accuracy']:.4f}")
            print(f"   Test Macro AUC: {test_metrics['auc_macro']:.4f}")
            
            # Generate evaluation plots
            print("📈 Generating evaluation plots...")
            generated_plots = evaluator.generate_all_plots(
                test_metrics, cv_results, data_processor.get_target_classes(),
                plots_dir, y_test, y_proba
            )
            print(f"   Generated {len(generated_plots)} plots: {generated_plots}")
            
            # Save individual artifacts
            print("💾 Saving model artifacts...")
            data_processor_path = os.path.join(model_artifacts_dir, "data_processor.pkl")
            feature_processor_path = os.path.join(model_artifacts_dir, "feature_processor.pkl") 
            model_wrapper_path = os.path.join(model_artifacts_dir, "trained_models.pkl")
            
            data_processor.save(data_processor_path)
            feature_processor.save(feature_processor_path)
            model_wrapper.save(model_wrapper_path)
            
            artifact_files = ["data_processor.pkl", "feature_processor.pkl", "trained_models.pkl"]
            
            # Create and test ModelPipeline
            print("🔗 Creating ModelPipeline...")
            pipeline = ModelPipeline(
                data_processor=data_processor,
                feature_processor=feature_processor, 
                model_wrapper=model_wrapper
            )
            
            # Test pipeline with sample data
            sample_input = X_test.iloc[:3]  # Use first 3 test samples
            sample_predictions = pipeline.predict(sample_input)
            sample_probabilities = pipeline.predict_proba(sample_input)
            print(f"   Pipeline test successful. Sample predictions: {sample_predictions}")
            
            # Create MLflow model
            print("🏗️ Creating MLflow model...")
            mlflow_model_dir = os.path.join(model_artifacts_dir, "mlflow_model")
            relative_mlflow_path = "output/model_artifacts/mlflow_model/"
            
            # Create signature
            signature = mlflow.models.infer_signature(sample_input, sample_predictions)
            
            # Save MLflow model locally
            print(f"💾 Saving model to local disk for harness: {mlflow_model_dir}")
            mlflow.sklearn.save_model(
                pipeline,
                path=mlflow_model_dir,
                signature=signature
            )
            
            # Log to MLflow if run ID available
            active_run_id = "fa8be7477dad42bf92df565c71b5ebd9"
            logged_model_uri = None
            
            if active_run_id and active_run_id != 'None' and active_run_id.strip():
                print(f"✅ Active MLflow run ID '{active_run_id}' detected. Reconnecting to log model as an artifact.")
                try:
                    with mlflow.start_run(run_id=active_run_id):
                        logged_model_info = mlflow.sklearn.log_model(
                            pipeline,
                            artifact_path="model",
                            code_paths=["iris_wandml_project"],
                            signature=signature
                        )
                        logged_model_uri = logged_model_info.model_uri
                        print(f"   Model logged with URI: {logged_model_uri}")
                except Exception as e:
                    print(f"⚠️ Failed to log to MLflow run: {e}")
            else:
                print("ℹ️ No active MLflow run ID provided. Skipping model logging.")
            
            # Add mlflow model to artifacts list
            artifact_files.append("mlflow_model/")
            
            # Save experiment metadata
            experiment_metadata = {
                "config": {
                    "data_prep": {
                        "version": self._config.data_prep.version,
                        "dataset_name": self._config.data_prep.dataset_name,
                        "feature_columns": self._config.data_prep.feature_columns,
                        "target_column": self._config.data_prep.target_column
                    },
                    "model": {
                        "model_type": self._config.model.model_type,
                        "model_params": self._config.model.model_params
                    }
                },
                "results": {
                    "cv_accuracy_mean": float(cv_results['cv_accuracy_mean']),
                    "cv_auc_mean": float(cv_results['cv_auc_mean']),
                    "test_accuracy": float(test_metrics['accuracy']),
                    "test_auc_macro": float(test_metrics['auc_macro'])
                },
                "generated_plots": generated_plots
            }
            
            metadata_path = os.path.join(general_artifacts_dir, "experiment_metadata.json")
            with open(metadata_path, 'w') as f:
                json.dump(experiment_metadata, f, indent=2)
            
            print("✅ Experiment completed successfully!")
            
            # Return results in required format
            return {
                "metric_name": "auc_macro",
                "metric_value": float(test_metrics['auc_macro']),
                "model_artifacts": artifact_files,
                "mlflow_model_info": {
                    "model_path": relative_mlflow_path,
                    "logged_model_uri": logged_model_uri,
                    "model_type": "sklearn",
                    "task_type": "classification",
                    "signature": signature.to_dict() if signature else None,
                    "input_example": sample_input.to_dict('records'),
                    "framework_version": sklearn.__version__
                }
            }
            
        except Exception as e:
            print(f"❌ Experiment failed with error: {str(e)}")
            raise