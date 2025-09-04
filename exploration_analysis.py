import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import label_binarize
import warnings
warnings.filterwarnings('ignore')

# Load data
train_data = pd.read_csv('/Users/yuvalheffetz/ds-agent-projects/session_3e0a615d-9170-492c-b8ab-7121cbd38bd2/data/train_set.csv')
test_data = pd.read_csv('/Users/yuvalheffetz/ds-agent-projects/session_3e0a615d-9170-492c-b8ab-7121cbd38bd2/data/test_set.csv')

print("=== Dataset Overview ===")
print(f"Training set shape: {train_data.shape}")
print(f"Test set shape: {test_data.shape}")
print(f"Class distribution in training set:")
print(train_data['Species'].value_counts())

# Prepare features and target
feature_cols = ['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']
X_train = train_data[feature_cols]
y_train = train_data['Species']
X_test = test_data[feature_cols]
y_test = test_data['Species']

# Encode labels
le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_test_encoded = le.transform(y_test)

print("\n=== Feature Statistics ===")
print(X_train.describe())

print("\n=== Correlation Matrix ===")
correlation_matrix = X_train.corr()
print(correlation_matrix)

# Test different preprocessing approaches
print("\n=== EXPERIMENT 1: Raw Features vs Standardized Features ===")

models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'SVM': SVC(random_state=42, probability=True),
    'Naive Bayes': GaussianNB()
}

# Test with raw features
print("\n--- Raw Features ---")
raw_results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train_encoded, cv=5, scoring='accuracy')
    raw_results[name] = cv_scores.mean()
    print(f"{name}: CV Accuracy = {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# Test with standardized features
print("\n--- Standardized Features ---")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

scaled_results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_scaled, y_train_encoded, cv=5, scoring='accuracy')
    scaled_results[name] = cv_scores.mean()
    print(f"{name}: CV Accuracy = {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

print("\n=== EXPERIMENT 2: Feature Engineering ===")

# Create new features based on domain knowledge
X_train_engineered = X_train.copy()
X_test_engineered = X_test.copy()

# Ratios and combinations that might be meaningful
X_train_engineered['sepal_ratio'] = X_train['SepalLengthCm'] / X_train['SepalWidthCm']
X_train_engineered['petal_ratio'] = X_train['PetalLengthCm'] / X_train['PetalWidthCm']
X_train_engineered['sepal_area'] = X_train['SepalLengthCm'] * X_train['SepalWidthCm']
X_train_engineered['petal_area'] = X_train['PetalLengthCm'] * X_train['PetalWidthCm']
X_train_engineered['total_length'] = X_train['SepalLengthCm'] + X_train['PetalLengthCm']
X_train_engineered['total_width'] = X_train['SepalWidthCm'] + X_train['PetalWidthCm']

X_test_engineered['sepal_ratio'] = X_test['SepalLengthCm'] / X_test['SepalWidthCm']
X_test_engineered['petal_ratio'] = X_test['PetalLengthCm'] / X_test['PetalWidthCm']
X_test_engineered['sepal_area'] = X_test['SepalLengthCm'] * X_test['SepalWidthCm']
X_test_engineered['petal_area'] = X_test['PetalLengthCm'] * X_test['PetalWidthCm']
X_test_engineered['total_length'] = X_test['SepalLengthCm'] + X_test['PetalLengthCm']
X_test_engineered['total_width'] = X_test['SepalWidthCm'] + X_test['PetalWidthCm']

# Scale engineered features
scaler_eng = StandardScaler()
X_train_eng_scaled = scaler_eng.fit_transform(X_train_engineered)

print("\n--- With Feature Engineering ---")
eng_results = {}
for name, model in models.items():
    cv_scores = cross_val_score(model, X_train_eng_scaled, y_train_encoded, cv=5, scoring='accuracy')
    eng_results[name] = cv_scores.mean()
    print(f"{name}: CV Accuracy = {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

print("\n=== EXPERIMENT 3: Model Complexity Analysis ===")

# Test different complexity levels for key algorithms
print("\n--- Random Forest: Different n_estimators ---")
for n_est in [10, 50, 100, 200]:
    rf = RandomForestClassifier(random_state=42, n_estimators=n_est)
    cv_scores = cross_val_score(rf, X_train_scaled, y_train_encoded, cv=5, scoring='accuracy')
    print(f"n_estimators={n_est}: CV Accuracy = {cv_scores.mean():.4f}")

print("\n--- Decision Tree: Different max_depth ---")
for depth in [3, 5, 10, None]:
    dt = DecisionTreeClassifier(random_state=42, max_depth=depth)
    cv_scores = cross_val_score(dt, X_train_scaled, y_train_encoded, cv=5, scoring='accuracy')
    print(f"max_depth={depth}: CV Accuracy = {cv_scores.mean():.4f}")

print("\n=== EXPERIMENT 4: Final Model Evaluation ===")

# Train best performing models on full training set and evaluate on test set
best_models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(random_state=42, n_estimators=100),
    'SVM': SVC(random_state=42, probability=True)
}

print("\n--- Test Set Performance ---")
for name, model in best_models.items():
    model.fit(X_train_scaled, y_train_encoded)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)
    
    accuracy = accuracy_score(y_test_encoded, y_pred)
    
    # Calculate macro-averaged AUC
    y_test_binarized = label_binarize(y_test_encoded, classes=[0, 1, 2])
    macro_auc = roc_auc_score(y_test_binarized, y_pred_proba, average='macro', multi_class='ovr')
    
    print(f"{name}:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Macro AUC: {macro_auc:.4f}")
    print(f"  Classification Report:")
    print(classification_report(y_test_encoded, y_pred, target_names=le.classes_))
    print()

print("\n=== EXPERIMENT 5: Feature Importance Analysis ===")

# Train Random Forest to get feature importance
rf_final = RandomForestClassifier(random_state=42, n_estimators=100)
rf_final.fit(X_train_scaled, y_train_encoded)

feature_importance = pd.DataFrame({
    'feature': feature_cols,
    'importance': rf_final.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature Importance (Random Forest):")
print(feature_importance)

# Logistic Regression coefficients
lr_final = LogisticRegression(random_state=42, max_iter=1000)
lr_final.fit(X_train_scaled, y_train_encoded)

print("\nLogistic Regression Coefficients:")
for i, class_name in enumerate(le.classes_):
    print(f"Class {class_name}:")
    coef_df = pd.DataFrame({
        'feature': feature_cols,
        'coefficient': lr_final.coef_[i]
    }).sort_values('coefficient', key=abs, ascending=False)
    print(coef_df)
    print()

print("\n=== SUMMARY OF FINDINGS ===")
print("1. Raw vs Standardized Features:")
for name in models.keys():
    print(f"   {name}: Raw={raw_results[name]:.4f}, Scaled={scaled_results[name]:.4f}")

print("\n2. Feature Engineering Impact:")
for name in models.keys():
    improvement = eng_results[name] - scaled_results[name]
    print(f"   {name}: Improvement = {improvement:.4f}")

print("\n3. Best performing algorithms consistently across experiments:")
top_algorithms = []
for name in models.keys():
    avg_performance = (raw_results[name] + scaled_results[name] + eng_results[name]) / 3
    top_algorithms.append((name, avg_performance))

top_algorithms.sort(key=lambda x: x[1], reverse=True)
for name, avg_perf in top_algorithms[:3]:
    print(f"   {name}: Average CV Accuracy = {avg_perf:.4f}")