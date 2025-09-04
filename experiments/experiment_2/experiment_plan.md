# Experiment 2: Multi-Algorithm Comparison with Hyperparameter Optimization

## Overview
**Experiment ID:** experiment_2  
**Iteration:** 2  
**Primary Change:** Implement comprehensive multi-algorithm comparison with hyperparameter optimization as originally planned in Iteration 1 but not executed

## Context & Rationale

### Previous Iteration Results
- **Iteration 1** achieved exceptional performance (99.7% macro AUC, 96.7% accuracy) using only SVM with RBF kernel
- **Gap identified:** Only 1 of 3 planned algorithms was implemented
- **Opportunity:** Complete the comprehensive model comparison to validate SVM superiority and explore potential improvements

### Exploration Findings
- **Linear SVM** achieved perfect 100% AUC in exploration
- **All algorithms** performed exceptionally well (>95% accuracy)
- **Hyperparameter optimization** showed potential for perfect scores
- **Dataset simplicity** confirmed, but algorithm comparison still valuable for completeness

## Data Preprocessing

### Data Loading
- **Train set:** `/Users/yuvalheffetz/ds-agent-projects/session_3e0a615d-9170-492c-b8ab-7121cbd38bd2/data/train_set.csv`
- **Test set:** `/Users/yuvalheffetz/ds-agent-projects/session_3e0a615d-9170-492c-b8ab-7121cbd38bd2/data/test_set.csv`

### Feature Processing
1. **Feature Selection:**
   - Use columns: `SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`
   - Drop column: `Id` (identifier only)

2. **Target Encoding:**
   - Apply `LabelEncoder` to `Species` column
   - Classes: Iris-setosa, Iris-versicolor, Iris-virginica

3. **Feature Scaling:**
   - Apply `StandardScaler` to all 4 numerical features
   - Critical for SVM and Logistic Regression performance

## Feature Engineering

**Approach:** Use original botanical features without transformation

**Rationale:** 
- Exploration experiments showed polynomial features provide no performance benefit
- Original 4 features are already highly discriminative with clear biological meaning
- Maintaining interpretability for botanical classification

**Final Features:** `SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`

## Model Selection Strategy

### Algorithms to Compare

#### 1. SVM (Linear Kernel)
- **Class:** `sklearn.svm.SVC`
- **Base Parameters:** `kernel='linear', probability=True, random_state=42`
- **Hyperparameter Grid:**
  - `C`: [0.1, 1, 10, 100]
- **Rationale:** Exploration showed perfect AUC performance. Test regularization impact.

#### 2. SVM (RBF Kernel) 
- **Class:** `sklearn.svm.SVC`
- **Base Parameters:** `kernel='rbf', probability=True, random_state=42`
- **Hyperparameter Grid:**
  - `C`: [0.1, 1, 10]
  - `gamma`: ['scale', 0.001, 0.01, 0.1]
- **Rationale:** Baseline model from Iteration 1. Optimize to improve upon 99.7% AUC.

#### 3. Logistic Regression
- **Class:** `sklearn.linear_model.LogisticRegression`
- **Base Parameters:** `random_state=42, max_iter=1000, multi_class='ovr'`
- **Hyperparameter Grid:**
  - `C`: [0.01, 0.1, 1, 10]
  - `solver`: ['liblinear', 'lbfgs']
- **Rationale:** Strong performance (99.8% AUC) with interpretable coefficients.

#### 4. Random Forest
- **Class:** `sklearn.ensemble.RandomForestClassifier`
- **Base Parameters:** `random_state=42`
- **Hyperparameter Grid:**
  - `n_estimators`: [50, 100, 200]
  - `max_depth`: [3, 5, None]
  - `min_samples_split`: [2, 5]
- **Rationale:** Ensemble method with built-in feature importance capabilities.

### Selection Process
1. **Cross-Validation:** 5-fold StratifiedKFold (shuffle=True, random_state=42)
2. **Hyperparameter Optimization:** GridSearchCV with 3-fold CV, scoring='roc_auc_ovr'
3. **Final Selection:** Best macro-averaged AUC from optimization

## Evaluation Strategy

### Primary Metric
**Macro-averaged AUC** (≥ 0.997 to match Iteration 1 performance)

### Secondary Metrics
- Test accuracy (≥ 0.967)
- Precision (macro-averaged)
- Recall (macro-averaged) 
- F1-score (macro-averaged)

### Evaluation Components

#### 1. Cross-Validation Performance Comparison
- Compare all 4 algorithms using 5-fold stratified CV
- Assess consistency and variance across folds
- **Output:** Algorithm performance ranking with confidence intervals

#### 2. Hyperparameter Optimization Analysis
- Document best parameters for each algorithm
- Quantify performance improvement over default parameters
- **Output:** Hyperparameter heatmaps and improvement metrics

#### 3. Test Set Performance Evaluation
- Evaluate optimized models on holdout test set
- Compare against Iteration 1 baseline
- **Output:** Final performance comparison and confusion matrices

#### 4. Algorithm Comparison Analysis
- **Performance ranking:** AUC and accuracy comparison
- **Training time analysis:** Computational efficiency comparison
- **Model complexity:** Parameter count and interpretability
- **Output:** Comprehensive algorithm strengths/weaknesses report

#### 5. Feature Importance Analysis
- Extract feature importance from Random Forest
- Analyze Logistic Regression coefficients
- **Output:** Feature importance comparison across model types

#### 6. Per-Class Performance Analysis
- Break down performance by iris species
- Identify class-specific patterns across algorithms
- **Output:** Per-class precision/recall/F1 analysis

## Expected Deliverables

### Model Artifacts
- `trained_models.pkl` - All 4 optimized models
- `best_hyperparameters.json` - Optimal parameters for each algorithm
- `data_processor.pkl` - Preprocessing pipeline
- `feature_processor.pkl` - Feature scaling pipeline

### Performance Reports
- `algorithm_comparison_summary.json` - Key metrics comparison
- `hyperparameter_optimization_results.json` - Tuning results
- `cross_validation_detailed_results.json` - Fold-by-fold performance

### Visualizations
- `algorithm_performance_comparison.html` - Side-by-side metrics
- `hyperparameter_heatmaps.html` - Parameter sensitivity for each model
- `confusion_matrices_all_models.html` - Classification patterns
- `roc_curves_comparison.html` - ROC curves for all algorithms
- `feature_importance_comparison.html` - Importance rankings
- `training_time_analysis.html` - Computational efficiency

### Analysis Reports
- `model_selection_rationale.md` - Algorithm choice justification
- `performance_vs_iteration1_comparison.md` - Baseline comparison
- `algorithm_strengths_weaknesses.md` - Detailed algorithm analysis

## Success Criteria

### Primary Success
- **Macro-averaged AUC ≥ 0.997** (match Iteration 1 baseline)
- **Complete implementation** of all 4 planned algorithms
- **Hyperparameter optimization** showing improvement for ≥2 algorithms

### Secondary Success  
- **Test accuracy ≥ 0.967** (match Iteration 1 baseline)
- **Comprehensive comparison** with actionable insights
- **Feature importance** analysis providing botanical insights

## Implementation Guidelines

### Technical Requirements
- Use `random_state=42` across all models for reproducibility
- Apply identical preprocessing pipeline to ensure fair comparison
- Use `roc_auc_ovr` scoring for multiclass AUC calculation
- Save all trained models for potential future ensemble methods

### Key Differences from Iteration 1
1. **4 algorithms instead of 1:** Complete the planned comparison
2. **Hyperparameter optimization:** GridSearchCV for all models  
3. **Enhanced evaluation:** Detailed algorithm comparison analysis
4. **Feature importance:** Extract insights where supported by models

### Expected Runtime
- **Longer than Iteration 1** due to 4 algorithms × hyperparameter grids
- **Justified by completeness:** Essential for thorough model comparison

## Next Iteration Considerations

Based on this comprehensive comparison, future iterations could explore:
- **Ensemble methods** combining top performers
- **Feature engineering** if any algorithm shows sensitivity
- **More challenging datasets** if all algorithms achieve perfect performance
- **Advanced techniques** (neural networks, gradient boosting) if linear methods plateau