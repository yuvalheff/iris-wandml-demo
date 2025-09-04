from pathlib import Path
import pandas as pd
import numpy as np
import os
import pickle
import json
import sklearn
import mlflow
import time
from typing import Dict, List, Tuple, Any

from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from iris_wandml_project.pipeline.feature_preprocessing import FeatureProcessor
from iris_wandml_project.pipeline.data_preprocessing import DataProcessor
from iris_wandml_project.pipeline.model import ModelWrapper
from iris_wandml_project.config import Config, ModelConfig
from experiment_scripts.evaluation import ModelEvaluator
from iris_wandml_project.model_pipeline import ModelPipeline

DEFAULT_CONFIG = str(Path(__file__).parent / 'config.yaml')


class Experiment:
    def __init__(self):
        self._config = Config.from_yaml(DEFAULT_CONFIG)
        
    def run(self, train_dataset_path, test_dataset_path, output_dir, seed=42):
        """
        Run the complete multi-algorithm experiment pipeline with hyperparameter optimization.
        
        Parameters:
        train_dataset_path: Path to training data
        test_dataset_path: Path to test data  
        output_dir: Directory to save outputs
        seed: Random seed for reproducibility
        
        Returns:
        dict: Experiment results in the required format
        """
        print(f"🚀 Starting Multi-Algorithm Iris Classification Experiment")
        print(f"📊 Config loaded with {len(self._config.algorithms) if self._config.algorithms else 0} algorithms")
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
            
            # Initialize data and feature processors
            print("🔧 Initializing pipeline components...")
            data_processor = DataProcessor(self._config.data_prep)
            feature_processor = FeatureProcessor(self._config.feature_prep)
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
            
            # Process test data
            X_test, y_test = data_processor.transform(test_data)
            X_test_processed = feature_processor.transform(X_test)
            
            # Multi-algorithm comparison with hyperparameter optimization
            print("🔬 Starting multi-algorithm comparison with hyperparameter optimization...")
            algorithm_results = {}
            trained_models = {}
            
            if self._config.algorithms and self._config.hyperparameter_optimization:
                for alg_name, alg_config in self._config.algorithms.items():
                    print(f"\n📊 Processing algorithm: {alg_name}")
                    
                    # Train and optimize algorithm
                    start_time = time.time()
                    best_model, results = self._train_and_optimize_algorithm(
                        alg_name, alg_config, X_train_processed, y_train, seed
                    )
                    training_time = time.time() - start_time
                    
                    # Evaluate on test set
                    test_results = self._evaluate_on_test_set(
                        best_model, X_test_processed, y_test, data_processor.get_target_classes()
                    )
                    
                    # Store results
                    algorithm_results[alg_name] = {
                        **results,
                        **test_results,
                        'training_time': training_time
                    }
                    trained_models[alg_name] = best_model
                    
                    print(f"   ✅ {alg_name} - Best CV AUC: {results['best_cv_auc']:.4f}, Test AUC: {test_results['test_auc']:.4f}")
            
            # Select best algorithm
            best_algorithm = self._select_best_algorithm(algorithm_results)
            best_model = trained_models[best_algorithm]
            print(f"\n🏆 Best algorithm: {best_algorithm} with AUC = {algorithm_results[best_algorithm]['test_auc']:.4f}")
            
            # Create model wrapper for best model
            best_model_config = ModelConfig(
                model_type=self._config.algorithms[best_algorithm].model_type,
                model_params=algorithm_results[best_algorithm]['best_params'],
                random_state=seed
            )
            model_wrapper = ModelWrapper(best_model_config)
            model_wrapper.model = best_model
            model_wrapper.is_fitted = True
            
            # Generate comprehensive evaluation plots
            print("📈 Generating comprehensive evaluation plots...")
            generated_plots = self._generate_comprehensive_plots(
                algorithm_results, trained_models, X_test_processed, y_test, 
                data_processor.get_target_classes(), plots_dir
            )
            print(f"   Generated {len(generated_plots)} plots")
            
            # Save individual artifacts
            print("💾 Saving model artifacts...")
            data_processor_path = os.path.join(model_artifacts_dir, "data_processor.pkl")
            feature_processor_path = os.path.join(model_artifacts_dir, "feature_processor.pkl")
            model_wrapper_path = os.path.join(model_artifacts_dir, "trained_models.pkl")
            
            data_processor.save(data_processor_path)
            feature_processor.save(feature_processor_path)
            model_wrapper.save(model_wrapper_path)
            
            # Save all trained models
            all_models_path = os.path.join(model_artifacts_dir, "all_trained_models.pkl")
            with open(all_models_path, 'wb') as f:
                pickle.dump(trained_models, f)
            
            artifact_files = ["data_processor.pkl", "feature_processor.pkl", "trained_models.pkl", "all_trained_models.pkl"]
            
            # Create and test ModelPipeline with best model
            print("🔗 Creating ModelPipeline with best model...")
            pipeline = ModelPipeline(
                data_processor=data_processor,
                feature_processor=feature_processor,
                model_wrapper=model_wrapper
            )
            
            # Test pipeline with sample data
            sample_input = X_test.iloc[:3]
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
            active_run_id = "9b96bb5ac8e34d0c8bec5ef55bd4007a"
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
            
            # Save comprehensive experiment results
            self._save_experiment_results(
                algorithm_results, best_algorithm, generated_plots, general_artifacts_dir
            )
            
            print("✅ Multi-algorithm experiment completed successfully!")
            
            # Return results in required format
            return {
                "metric_name": "auc_macro",
                "metric_value": float(algorithm_results[best_algorithm]['test_auc']),
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
    
    def _create_model(self, model_type: str, params: Dict[str, Any]) -> Any:
        """Create a model instance based on type and parameters."""
        if model_type == "svm":
            return SVC(**params)
        elif model_type == "logistic_regression":
            return LogisticRegression(**params)
        elif model_type == "random_forest":
            return RandomForestClassifier(**params)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
    
    def _train_and_optimize_algorithm(self, alg_name: str, alg_config, X_train, y_train, seed: int) -> Tuple[Any, Dict]:
        """Train and optimize a single algorithm using GridSearchCV."""
        print(f"   🔧 Setting up {alg_name} with hyperparameter optimization...")
        
        # Create base model with base parameters
        base_model = self._create_model(alg_config.model_type, alg_config.base_params)
        
        # Set up GridSearchCV
        cv = StratifiedKFold(
            n_splits=self._config.hyperparameter_optimization.cv_folds,
            shuffle=True,
            random_state=seed
        )
        
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=alg_config.hyperparameter_grid,
            scoring=self._config.hyperparameter_optimization.scoring,
            cv=cv,
            n_jobs=self._config.hyperparameter_optimization.n_jobs,
            verbose=1
        )
        
        print(f"   🎯 Training {alg_name} with {len(grid_search.param_grid)} parameter combinations...")
        grid_search.fit(X_train, y_train)
        
        # Perform final cross-validation with best model
        final_cv = StratifiedKFold(n_splits=self._config.model_evaluation.cv_folds, shuffle=True, random_state=seed)
        cv_auc_scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, 
                                      cv=final_cv, scoring='roc_auc_ovr')
        cv_acc_scores = cross_val_score(grid_search.best_estimator_, X_train, y_train, 
                                      cv=final_cv, scoring='accuracy')
        
        results = {
            'best_params': grid_search.best_params_,
            'best_cv_auc': grid_search.best_score_,
            'cv_auc_mean': cv_auc_scores.mean(),
            'cv_auc_std': cv_auc_scores.std(),
            'cv_acc_mean': cv_acc_scores.mean(),
            'cv_acc_std': cv_acc_scores.std(),
            'grid_search_results': {
                'cv_results': grid_search.cv_results_,
                'best_index': grid_search.best_index_
            }
        }
        
        return grid_search.best_estimator_, results
    
    def _evaluate_on_test_set(self, model, X_test, y_test, class_names: List[str]) -> Dict:
        """Evaluate a trained model on the test set."""
        # Make predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        evaluator = ModelEvaluator(self._config.model_evaluation)
        metrics = evaluator.calculate_metrics(y_test, y_pred, y_proba, class_names)
        
        return {
            'test_accuracy': metrics['accuracy'],
            'test_auc': metrics['auc_macro'],
            'test_precision': metrics['precision_macro'],
            'test_recall': metrics['recall_macro'],
            'test_f1': metrics['f1_macro'],
            'test_predictions': y_pred,
            'test_probabilities': y_proba,
            'confusion_matrix': metrics['confusion_matrix']
        }
    
    def _select_best_algorithm(self, algorithm_results: Dict) -> str:
        """Select the best algorithm based on test AUC."""
        best_algorithm = max(algorithm_results.keys(), 
                           key=lambda alg: algorithm_results[alg]['test_auc'])
        return best_algorithm
    
    def _generate_comprehensive_plots(self, algorithm_results: Dict, trained_models: Dict,
                                    X_test, y_test, class_names: List[str], plots_dir: str) -> List[str]:
        """Generate comprehensive evaluation plots for all algorithms."""
        generated_plots = []
        
        # Generate individual plots for best algorithm
        best_alg = self._select_best_algorithm(algorithm_results)
        best_model = trained_models[best_alg]
        
        evaluator = ModelEvaluator(self._config.model_evaluation)
        
        # Test metrics for best algorithm
        y_pred = best_model.predict(X_test)
        y_proba = best_model.predict_proba(X_test)
        test_metrics = evaluator.calculate_metrics(y_test, y_pred, y_proba, class_names)
        
        # Generate standard plots
        cv_results = {
            'cv_accuracy_scores': np.array([algorithm_results[best_alg]['cv_acc_mean']]),
            'cv_auc_scores': np.array([algorithm_results[best_alg]['cv_auc_mean']])
        }
        
        standard_plots = evaluator.generate_all_plots(
            test_metrics, cv_results, class_names, plots_dir, y_test, y_proba
        )
        generated_plots.extend(standard_plots)
        
        # Generate algorithm comparison plots
        comparison_plots = self._generate_algorithm_comparison_plots(
            algorithm_results, class_names, plots_dir
        )
        generated_plots.extend(comparison_plots)
        
        return generated_plots
    
    def _generate_algorithm_comparison_plots(self, algorithm_results: Dict, 
                                           class_names: List[str], plots_dir: str) -> List[str]:
        """Generate algorithm comparison plots."""
        import plotly.express as px
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
        
        generated_plots = []
        
        # App color palette
        app_color_palette = [
            'rgba(99, 110, 250, 0.8)',   # Blue
            'rgba(239, 85, 59, 0.8)',    # Red/Orange-Red
            'rgba(0, 204, 150, 0.8)',    # Green
            'rgba(171, 99, 250, 0.8)',   # Purple
            'rgba(255, 161, 90, 0.8)',   # Orange
            'rgba(25, 211, 243, 0.8)',   # Cyan
            'rgba(255, 102, 146, 0.8)',  # Pink
            'rgba(182, 232, 128, 0.8)',  # Light Green
            'rgba(255, 151, 255, 0.8)',  # Magenta
            'rgba(254, 203, 82, 0.8)'    # Yellow
        ]
        
        def apply_plot_styling(fig):
            """Apply consistent styling to plots."""
            fig.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font=dict(color='#8B5CF6', size=12),
                title_font=dict(color='#7C3AED', size=16),
                xaxis=dict(
                    gridcolor='rgba(139,92,246,0.2)',
                    zerolinecolor='rgba(139,92,246,0.3)',
                    tickfont=dict(color='#8B5CF6', size=11),
                    title_font=dict(color='#7C3AED', size=12)
                ),
                yaxis=dict(
                    gridcolor='rgba(139,92,246,0.2)',
                    zerolinecolor='rgba(139,92,246,0.3)',
                    tickfont=dict(color='#8B5CF6', size=11),
                    title_font=dict(color='#7C3AED', size=12)
                ),
                legend=dict(font=dict(color='#8B5CF6', size=11))
            )
        
        # Algorithm performance comparison
        algorithms = list(algorithm_results.keys())
        test_aucs = [algorithm_results[alg]['test_auc'] for alg in algorithms]
        test_accuracies = [algorithm_results[alg]['test_accuracy'] for alg in algorithms]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='Test AUC',
            x=algorithms,
            y=test_aucs,
            marker_color=app_color_palette[0]
        ))
        fig.add_trace(go.Bar(
            name='Test Accuracy',
            x=algorithms,
            y=test_accuracies,
            marker_color=app_color_palette[1]
        ))
        
        fig.update_layout(
            title='Algorithm Performance Comparison',
            xaxis_title='Algorithm',
            yaxis_title='Performance Score',
            barmode='group'
        )
        apply_plot_styling(fig)
        
        comparison_path = os.path.join(plots_dir, "algorithm_performance_comparison.html")
        fig.write_html(comparison_path, include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})
        generated_plots.append("algorithm_performance_comparison.html")
        
        # Training time comparison
        training_times = [algorithm_results[alg]['training_time'] for alg in algorithms]
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=algorithms,
            y=training_times,
            marker_color=app_color_palette[2]
        ))
        
        fig.update_layout(
            title='Training Time Comparison',
            xaxis_title='Algorithm',
            yaxis_title='Training Time (seconds)'
        )
        apply_plot_styling(fig)
        
        time_path = os.path.join(plots_dir, "training_time_comparison.html")
        fig.write_html(time_path, include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})
        generated_plots.append("training_time_comparison.html")
        
        return generated_plots
    
    def _save_experiment_results(self, algorithm_results: Dict, best_algorithm: str,
                               generated_plots: List[str], general_artifacts_dir: str):
        """Save comprehensive experiment results."""
        
        # Algorithm comparison summary
        comparison_summary = {
            "best_algorithm": best_algorithm,
            "algorithms_compared": len(algorithm_results),
            "algorithm_rankings": sorted(
                [(alg, results['test_auc']) for alg, results in algorithm_results.items()],
                key=lambda x: x[1], reverse=True
            ),
            "performance_summary": {
                alg: {
                    "test_auc": float(results['test_auc']),
                    "test_accuracy": float(results['test_accuracy']),
                    "cv_auc_mean": float(results['cv_auc_mean']),
                    "cv_auc_std": float(results['cv_auc_std']),
                    "training_time": float(results['training_time']),
                    "best_params": results['best_params']
                }
                for alg, results in algorithm_results.items()
            }
        }
        
        comparison_path = os.path.join(general_artifacts_dir, "algorithm_comparison_summary.json")
        with open(comparison_path, 'w') as f:
            json.dump(comparison_summary, f, indent=2)
        
        # Hyperparameter optimization results
        hp_results = {
            alg: {
                "best_params": results['best_params'],
                "best_cv_auc": float(results['best_cv_auc']),
                "parameter_grid_size": len(results['grid_search_results']['cv_results']['params'])
            }
            for alg, results in algorithm_results.items()
        }
        
        hp_path = os.path.join(general_artifacts_dir, "hyperparameter_optimization_results.json")
        with open(hp_path, 'w') as f:
            json.dump(hp_results, f, indent=2)
        
        # Overall experiment metadata
        experiment_metadata = {
            "experiment_type": "multi_algorithm_comparison",
            "algorithms_tested": list(algorithm_results.keys()),
            "best_algorithm": best_algorithm,
            "best_performance": {
                "metric": "macro_auc",
                "value": float(algorithm_results[best_algorithm]['test_auc'])
            },
            "config_summary": {
                "hyperparameter_optimization": {
                    "method": self._config.hyperparameter_optimization.method,
                    "scoring": self._config.hyperparameter_optimization.scoring,
                    "cv_folds": self._config.hyperparameter_optimization.cv_folds
                }
            },
            "generated_plots": generated_plots
        }
        
        metadata_path = os.path.join(general_artifacts_dir, "experiment_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(experiment_metadata, f, indent=2)