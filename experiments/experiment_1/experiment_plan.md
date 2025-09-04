# Iris Multi-class Classification Baseline Experiment Plan

## Experiment Overview
**Experiment Name:** Iris Multi-class Classification Baseline  
**Task Type:** Multiclass Classification  
**Target Variable:** Species  
**Primary Metric:** Macro-averaged AUC  
**Dataset:** Classic Iris flower dataset (120 training samples, 30 test samples)

## Data Preprocessing Steps

### Step 1: Data Loading and Initial Processing
- Load training data from: `/Users/yuvalheffetz/ds-agent-projects/session_3e0a615d-9170-492c-b8ab-7121cbd38bd2/data/train_set.csv`
- Load test data from: `/Users/yuvalheffetz/ds-agent-projects/session_3e0a615d-9170-492c-b8ab-7121cbd38bd2/data/test_set.csv`
- Remove the `Id` column as it provides no predictive value

### Step 2: Feature and Target Extraction
- **Feature columns:** `['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']`
- **Target column:** `'Species'`
- Verify data types and check for any missing values (none expected based on EDA)

### Step 3: Target Encoding
- Apply `sklearn.preprocessing.LabelEncoder` to convert species names to numeric labels
- Transform: `'Iris-setosa'`, `'Iris-versicolor'`, `'Iris-virginica'` → `[0, 1, 2]`
- Save the encoder for inverse transformation during evaluation

### Step 4: Feature Standardization
- Apply `sklearn.preprocessing.StandardScaler` to normalize feature scales
- Fit scaler on training data and transform both training and test sets
- **Rationale:** While experiments showed minimal impact, standardization ensures algorithmic consistency

## Feature Engineering Strategy

### Approach: Minimal Feature Engineering
**Rationale:** Exploration experiments demonstrated that additional engineered features (ratios, areas, sums) provide no meaningful performance improvement over the original four botanical measurements.

### Implementation:
- **Use original features only:** Maintain `SepalLengthCm`, `SepalWidthCm`, `PetalLengthCm`, `PetalWidthCm`
- **No additional features:** Skip ratio calculations, area computations, or feature combinations
- **Focus on data quality:** Ensure proper scaling and encoding of existing high-quality features

## Model Selection Strategy

### Primary Algorithm: Support Vector Machine (SVM)
**Selection Rationale:** 
- Achieved highest test performance in exploration experiments (96.67% accuracy, 99.67% macro AUC)
- Excellent performance on linearly separable datasets like Iris
- Robust to small dataset sizes

**Hyperparameters:**
```python
SVC(kernel='rbf', random_state=42, probability=True)
```

### Backup Algorithms for Comparison:

#### 1. Logistic Regression
- **Performance:** 93.33% accuracy, 99.67% macro AUC in experiments
- **Benefits:** High interpretability, provides feature coefficients
- **Hyperparameters:** `LogisticRegression(random_state=42, max_iter=1000)`

#### 2. Random Forest
- **Performance:** 90% accuracy in experiments
- **Benefits:** Provides feature importance rankings, ensemble robustness  
- **Hyperparameters:** `RandomForestClassifier(n_estimators=100, random_state=42)`

## Evaluation Strategy

### Primary Metric: Macro-averaged AUC
- **Calculation:** Average AUC across all three classes using one-vs-rest approach
- **Target:** ≥ 0.95 based on exploration results

### Additional Metrics:
- **Accuracy:** Overall classification accuracy
- **Per-class Precision, Recall, F1-score:** Identify class-specific performance issues
- **Confusion Matrix:** Understand misclassification patterns

### Validation Approach:
- **Development:** 5-fold stratified cross-validation on training data
- **Final Evaluation:** Single evaluation on holdout test set (30 samples)

## Diagnostic Analysis Plan

### 1. Per-Class Performance Analysis
**Purpose:** Identify which species are most challenging to classify
**Implementation:**
- Generate detailed classification report with per-class metrics
- Compare performance across `Iris-setosa`, `Iris-versicolor`, `Iris-virginica`
- Flag any class with F1-score below 0.85

### 2. Confusion Matrix Analysis  
**Purpose:** Understand specific misclassification patterns
**Implementation:**
- Create normalized confusion matrix visualization
- Identify which species pairs are most commonly confused
- Analyze whether errors align with known botanical similarities

### 3. Feature Importance Analysis
**Purpose:** Understand which measurements are most discriminative
**Implementation:**
- Extract Random Forest feature importances for `PetalLengthCm`, `PetalWidthCm`, `SepalLengthCm`, `SepalWidthCm`
- Analyze Logistic Regression coefficients for each class
- Validate against EDA findings showing petal measurements as most important

### 4. Prediction Confidence Analysis
**Purpose:** Assess model certainty and identify edge cases
**Implementation:**
- Analyze distribution of maximum prediction probabilities
- Identify low-confidence predictions (max probability < 0.7)
- Examine characteristics of uncertain predictions

## Expected Outputs

### Model Artifacts:
- `trained_svm_model.pkl` - Primary SVM model
- `trained_logistic_model.pkl` - Logistic regression backup
- `trained_rf_model.pkl` - Random forest for feature importance
- `label_encoder.pkl` - Target label encoder
- `feature_scaler.pkl` - Feature standardization scaler

### Evaluation Reports:
- `model_performance_report.json` - Comprehensive performance metrics
- `confusion_matrices.png` - Visualization of classification errors
- `feature_importance_plot.png` - Feature contribution analysis
- `prediction_confidence_analysis.json` - Confidence distribution analysis

### Analysis Files:
- `per_class_analysis.json` - Detailed per-species performance breakdown
- `model_comparison_report.json` - Comparison across all three algorithms

## Success Criteria

### Performance Thresholds:
- **Macro-averaged AUC:** ≥ 0.95
- **Overall Accuracy:** ≥ 0.90  
- **Class Balance:** No class with F1-score < 0.85
- **Prediction Confidence:** ≥ 90% of predictions with confidence > 0.7

### Quality Requirements:
- Perfect classification of `Iris-setosa` (linearly separable class)
- Reasonable performance on `Iris-versicolor` vs `Iris-virginica` distinction
- Consistent performance across cross-validation folds (std < 0.05)

## Implementation Notes

### Data Quality Considerations:
- **Clean Dataset:** No missing values or significant outliers requiring treatment
- **Perfect Balance:** 40 samples per class eliminates need for class balancing techniques
- **Small Scale:** 120 training samples suitable for simple algorithms, avoid overly complex models

### Technical Specifications:
- **Reproducibility:** Set `random_state=42` for all stochastic algorithms
- **Computational Requirements:** Lightweight models appropriate for dataset size
- **Interpretability Priority:** Focus on interpretable models given classical benchmark nature

### Key Insights from Exploration:
- Original features provide optimal discrimination (no feature engineering needed)
- SVM consistently outperforms other algorithms on this dataset
- Petal measurements (`PetalLengthCm`, `PetalWidthCm`) are most discriminative features
- `Iris-setosa` is perfectly separable, main challenge is `Iris-versicolor` vs `Iris-virginica`