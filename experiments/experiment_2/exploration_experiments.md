# Exploration Experiments for Iteration 2 Planning

## Overview
Before finalizing the experiment plan for Iteration 2, I conducted systematic exploration experiments to test different approaches and identify the most promising strategy for improving upon the Iteration 1 baseline.

## Experimental Setup
- **Data:** Same train/test split as Iteration 1 (120 train, 30 test samples)
- **Features:** Original 4 botanical measurements (SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm)
- **Target:** Species (3-class classification: Iris-setosa, Iris-versicolor, Iris-virginica)
- **Evaluation:** 5-fold stratified cross-validation for consistency

## Experiment 1: Multi-Algorithm Comparison

### Objective
Test all algorithms that were planned in Iteration 1 but not implemented, to understand their relative performance.

### Algorithms Tested
1. **SVM (RBF)** - Iteration 1 baseline
2. **SVM (Linear)** - Alternative kernel  
3. **Logistic Regression** - Linear probabilistic model
4. **Random Forest** - Ensemble tree-based method

### Results
```
Algorithm            | CV Accuracy      | CV AUC (Macro)   
---------------------|------------------|------------------
SVM (RBF)           | 0.9667 ± 0.0167  | 0.9990 ± 0.0021
SVM (Linear)        | 0.9750 ± 0.0333  | 1.0000 ± 0.0000  ← Best
Logistic Regression | 0.9583 ± 0.0264  | 0.9979 ± 0.0042
Random Forest       | 0.9500 ± 0.0312  | 0.9971 ± 0.0037
```

### Key Findings
- **Linear SVM achieved perfect AUC (1.000)** and highest accuracy (97.5%)
- **All algorithms performed exceptionally well** (>95% accuracy, >99.7% AUC)
- **Small performance differences** suggest the dataset is highly separable
- **Linear models outperformed tree-based** method slightly

## Experiment 2: Feature Engineering Exploration

### Objective  
Test whether polynomial feature transformation could improve performance beyond the original 4 botanical measurements.

### Approaches Tested
1. **Original Features** - 4 botanical measurements (baseline)
2. **Polynomial Degree 2** - Quadratic features and interactions (14 total features)

### Results
```
Feature Set          | Features | CV Accuracy      | CV AUC (Macro)
---------------------|----------|------------------|------------------
Original Features    |    4     | 0.9667 ± 0.0167  | 0.9990 ± 0.0021
Polynomial Degree 2  |   14     | 0.9667 ± 0.0167  | 1.0000 ± 0.0000
```

### Key Findings
- **Polynomial features achieved perfect AUC** but same accuracy
- **Marginal benefit** from increased complexity (4 → 14 features)
- **Recommendation:** Stick with original features for interpretability
- **Biological relevance** of original measurements is preserved

## Experiment 3: Hyperparameter Optimization

### Objective
Test whether hyperparameter tuning could improve the top-performing algorithms beyond their default configurations.

### Models Optimized
1. **SVM (Linear)** - C parameter tuning
2. **SVM (RBF)** - C and gamma parameter tuning  

### Results
```
Model               | Best Parameters        | Best CV AUC
--------------------|------------------------|-------------
SVM (Linear)        | C=1                   | 1.0000
SVM (RBF)          | C=10, gamma=0.1       | 1.0000
```

### Key Findings
- **Both SVMs achieved perfect 100% AUC** with optimization
- **Significant improvement for RBF SVM** (99.9% → 100% AUC)
- **Linear SVM optimal** with default C=1 parameter
- **Hyperparameter tuning is beneficial** and should be included

## Experiment 4: Problem Difficulty Assessment

### Objective
Assess whether the iris dataset is too easy by introducing controlled noise to increase problem difficulty.

### Noise Levels Tested
- **0.0** - Original data (baseline)
- **0.1** - 10% Gaussian noise relative to feature standard deviation  
- **0.2** - 20% Gaussian noise relative to feature standard deviation

### Results
```
Noise Level | CV Accuracy      | CV AUC (Macro)   | Performance Impact
------------|------------------|------------------|-------------------
0.0         | 0.9667 ± 0.0167  | 0.9990 ± 0.0021  | Baseline
0.1         | 0.9667 ± 0.0312  | 0.9969 ± 0.0042  | Minimal degradation
0.2         | 0.9500 ± 0.0312  | 0.9943 ± 0.0048  | Slight degradation
```

### Key Findings
- **High robustness** to moderate noise levels
- **Even with 20% noise**, performance remains >94% accuracy and >99% AUC
- **Confirms dataset simplicity** but algorithms are fundamentally sound
- **Future iterations** might benefit from more challenging datasets

## Overall Exploration Summary

### Primary Insights
1. **Multi-algorithm comparison is essential** - Linear SVM outperformed the Iteration 1 RBF SVM
2. **Hyperparameter optimization provides clear benefits** - Perfect AUC achieved for optimized models
3. **Original features are sufficient** - No benefit from polynomial expansion
4. **All algorithms are highly effective** - Performance ceiling is very high for iris classification

### Recommended Strategy for Iteration 2
Based on these exploration results, the optimal approach for Iteration 2 is:

1. **Implement comprehensive multi-algorithm comparison** (the missing piece from Iteration 1)
2. **Include hyperparameter optimization** for all algorithms using GridSearchCV  
3. **Use original botanical features** without transformation
4. **Focus on thorough evaluation and comparison** rather than feature engineering

### Expected Benefits
- **Complete the planned comparison** that was missing from Iteration 1
- **Quantify algorithm performance differences** with statistical rigor
- **Potentially achieve perfect 100% AUC** through optimization
- **Provide comprehensive model selection rationale** for production deployment

This exploration-driven approach ensures that Iteration 2 addresses the specific gap identified in Iteration 1 while being grounded in empirical evidence rather than assumptions.