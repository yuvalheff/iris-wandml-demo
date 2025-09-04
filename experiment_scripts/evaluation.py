import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
from typing import Dict, List, Tuple

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, auc
)
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import cross_val_score, StratifiedKFold

from iris_wandml_project.config import ModelEvalConfig


class ModelEvaluator:
    def __init__(self, config: ModelEvalConfig):
        self.config: ModelEvalConfig = config
        
        # App color palette
        self.app_color_palette = [
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

    def calculate_metrics(self, y_true: pd.Series, y_pred: np.ndarray, 
                         y_proba: np.ndarray = None, class_names: List[str] = None) -> Dict:
        """
        Calculate evaluation metrics for multiclass classification.
        
        Parameters:
        y_true: True labels
        y_pred: Predicted labels  
        y_proba: Predicted probabilities (optional)
        class_names: List of class names (optional)
        
        Returns:
        Dict: Dictionary containing all evaluation metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        
        # Per-class metrics
        metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None)
        metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None)
        metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None)
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred)
        
        # AUC metrics (if probabilities are provided)
        if y_proba is not None:
            # Macro-averaged AUC (primary metric)
            metrics['auc_macro'] = roc_auc_score(y_true, y_proba, 
                                               multi_class='ovr', average='macro')
            
            # Per-class AUC
            metrics['auc_per_class'] = roc_auc_score(y_true, y_proba, 
                                                   multi_class='ovr', average=None)
        
        # Classification report
        if class_names:
            metrics['classification_report'] = classification_report(
                y_true, y_pred, target_names=class_names, output_dict=True
            )
        
        return metrics

    def cross_validate_model(self, model, X: pd.DataFrame, y: pd.Series) -> Dict:
        """
        Perform cross-validation on the model.
        
        Parameters:
        model: The model to evaluate
        X: Features
        y: Target
        
        Returns:
        Dict: Cross-validation results
        """
        cv = StratifiedKFold(n_splits=self.config.cv_folds, 
                           shuffle=True, random_state=self.config.random_state)
        
        # Accuracy scores
        cv_scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy')
        
        # AUC scores  
        auc_scores = cross_val_score(model, X, y, cv=cv, scoring='roc_auc_ovr')
        
        return {
            'cv_accuracy_scores': cv_scores,
            'cv_accuracy_mean': cv_scores.mean(),
            'cv_accuracy_std': cv_scores.std(),
            'cv_auc_scores': auc_scores,
            'cv_auc_mean': auc_scores.mean(),
            'cv_auc_std': auc_scores.std()
        }

    def generate_confusion_matrix_plot(self, confusion_mat: np.ndarray, 
                                     class_names: List[str], 
                                     output_path: str) -> None:
        """Generate confusion matrix heatmap plot."""
        fig = px.imshow(confusion_mat,
                       labels=dict(x="Predicted", y="Actual", color="Count"),
                       x=class_names,
                       y=class_names,
                       color_continuous_scale='Blues',
                       title="Confusion Matrix")
        
        # Add text annotations
        for i in range(len(class_names)):
            for j in range(len(class_names)):
                fig.add_annotation(x=j, y=i, text=str(confusion_mat[i, j]),
                                 showarrow=False, font=dict(color='white' if confusion_mat[i, j] > confusion_mat.max()/2 else 'black'))
        
        self._apply_plot_styling(fig)
        fig.write_html(output_path, include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

    def generate_roc_curve_plot(self, y_true: pd.Series, y_proba: np.ndarray, 
                              class_names: List[str], output_path: str) -> None:
        """Generate ROC curves for multiclass classification."""
        n_classes = len(class_names)
        
        # Binarize the labels
        y_bin = label_binarize(y_true, classes=np.unique(y_true))
        
        # Create subplots
        fig = make_subplots(rows=1, cols=1, 
                           subplot_titles=["ROC Curves - One-vs-Rest"])
        
        # Calculate ROC curve for each class
        for i, class_name in enumerate(class_names):
            fpr, tpr, _ = roc_curve(y_bin[:, i], y_proba[:, i])
            roc_auc = auc(fpr, tpr)
            
            fig.add_trace(go.Scatter(x=fpr, y=tpr,
                                   mode='lines',
                                   name=f'{class_name} (AUC = {roc_auc:.3f})',
                                   line=dict(color=self.app_color_palette[i % len(self.app_color_palette)])))
        
        # Add diagonal line
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1],
                               mode='lines',
                               name='Random Classifier',
                               line=dict(dash='dash', color='gray')))
        
        fig.update_layout(
            title='ROC Curves - Multi-class Classification',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            showlegend=True
        )
        
        self._apply_plot_styling(fig)
        fig.write_html(output_path, include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

    def generate_metrics_summary_plot(self, metrics: Dict, class_names: List[str], 
                                    output_path: str) -> None:
        """Generate a summary plot of key metrics."""
        # Prepare data for plotting
        metric_names = ['Precision', 'Recall', 'F1-Score']
        per_class_metrics = [
            metrics['precision_per_class'],
            metrics['recall_per_class'], 
            metrics['f1_per_class']
        ]
        
        fig = go.Figure()
        
        # Add bars for each class
        for i, class_name in enumerate(class_names):
            values = [metric[i] for metric in per_class_metrics]
            fig.add_trace(go.Bar(
                name=class_name,
                x=metric_names,
                y=values,
                marker_color=self.app_color_palette[i % len(self.app_color_palette)]
            ))
        
        fig.update_layout(
            title='Per-Class Performance Metrics',
            xaxis_title='Metrics',
            yaxis_title='Score',
            barmode='group',
            showlegend=True
        )
        
        self._apply_plot_styling(fig)
        fig.write_html(output_path, include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

    def generate_cv_results_plot(self, cv_results: Dict, output_path: str) -> None:
        """Generate cross-validation results visualization."""
        fig = make_subplots(rows=1, cols=2,
                           subplot_titles=['CV Accuracy Scores', 'CV AUC Scores'])
        
        # Accuracy scores
        fig.add_trace(go.Box(y=cv_results['cv_accuracy_scores'],
                           name='Accuracy',
                           marker_color=self.app_color_palette[0]), row=1, col=1)
        
        # AUC scores  
        fig.add_trace(go.Box(y=cv_results['cv_auc_scores'],
                           name='AUC',
                           marker_color=self.app_color_palette[1]), row=1, col=2)
        
        fig.update_layout(
            title='Cross-Validation Results',
            showlegend=False
        )
        
        self._apply_plot_styling(fig)
        fig.write_html(output_path, include_plotlyjs=True, config={'responsive': True, 'displayModeBar': False})

    def generate_all_plots(self, metrics: Dict, cv_results: Dict, 
                          class_names: List[str], output_dir: str, 
                          y_true: pd.Series = None, y_proba: np.ndarray = None) -> List[str]:
        """
        Generate all evaluation plots.
        
        Parameters:
        metrics: Dictionary of calculated metrics
        cv_results: Cross-validation results
        class_names: List of class names
        output_dir: Directory to save plots
        y_true: True labels (for ROC curve)
        y_proba: Predicted probabilities (for ROC curve)
        
        Returns:
        List[str]: List of generated plot file paths
        """
        os.makedirs(output_dir, exist_ok=True)
        generated_plots = []
        
        # Confusion Matrix
        cm_path = os.path.join(output_dir, "confusion_matrix.html")
        self.generate_confusion_matrix_plot(metrics['confusion_matrix'], class_names, cm_path)
        generated_plots.append("confusion_matrix.html")
        
        # ROC Curves (if probabilities available)
        if y_true is not None and y_proba is not None:
            roc_path = os.path.join(output_dir, "roc_curves.html")
            self.generate_roc_curve_plot(y_true, y_proba, class_names, roc_path)
            generated_plots.append("roc_curves.html")
        
        # Metrics Summary
        metrics_path = os.path.join(output_dir, "metrics_summary.html")
        self.generate_metrics_summary_plot(metrics, class_names, metrics_path)
        generated_plots.append("metrics_summary.html")
        
        # CV Results
        cv_path = os.path.join(output_dir, "cv_results.html")
        self.generate_cv_results_plot(cv_results, cv_path)
        generated_plots.append("cv_results.html")
        
        return generated_plots

    def _apply_plot_styling(self, fig) -> None:
        """Apply consistent styling to plots."""
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',  # Transparent background
            plot_bgcolor='rgba(0,0,0,0)',   # Transparent plot area
            font=dict(color='#8B5CF6', size=12),  # App's purple color for text
            title_font=dict(color='#7C3AED', size=16),  # Slightly darker purple for titles
            xaxis=dict(
                gridcolor='rgba(139,92,246,0.2)',  # Purple-tinted grid
                zerolinecolor='rgba(139,92,246,0.3)',
                tickfont=dict(color='#8B5CF6', size=11),  # Purple tick labels
                title_font=dict(color='#7C3AED', size=12)  # Darker purple axis titles
            ),
            yaxis=dict(
                gridcolor='rgba(139,92,246,0.2)',  # Purple-tinted grid
                zerolinecolor='rgba(139,92,246,0.3)', 
                tickfont=dict(color='#8B5CF6', size=11),  # Purple tick labels
                title_font=dict(color='#7C3AED', size=12)  # Darker purple axis titles
            ),
            legend=dict(font=dict(color='#8B5CF6', size=11))  # Purple legend
        )
