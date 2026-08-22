# Credit Card Fraud Detection Project - Step-by-Step Instructions

## How to Approach This Project

This guide walks you through each phase with concrete steps, code snippets, and explanations.

---

## PHASE 1: SETUP & DATA LOADING (30 minutes)

### Step 1.1: Create Project Directory and Download Dataset

```bash
# Create project folder
mkdir fraud_detection_project
cd fraud_detection_project

# Download dataset from Kaggle
# Option 1: Download manually from https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
# Option 2: Use Kaggle CLI (if installed)
# Option 3: Use provided dataset in course materials
```

### Step 1.2: Create Jupyter Notebook and Import Libraries

Create a file named `fraud_detection.ipynb` and start with:

```python
# SECTION 1: IMPORTS AND SETUP
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, classification_report, 
                             roc_curve, auc, roc_auc_score, f1_score, 
                             precision_score, recall_score, ConfusionMatrixDisplay)
import warnings
warnings.filterwarnings('ignore')

# Set visualization defaults
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
%matplotlib inline
```

### Step 1.3: Load the Dataset

```python
# SECTION 1: DATA LOADING
# Load the credit card fraud dataset
df = pd.read_csv('creditcard.csv')

print("Dataset Shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

print("\nDataset Info:")
print(df.info())

print("\nBasic Statistics:")
print(df.describe())
```

**Expected Output:**
- Shape: (284807, 31)
- Columns: Time, Amount, V1-V28, Class
- Class distribution: Highly imbalanced (0.17% fraud)

---

## PHASE 2: EXPLORATORY DATA ANALYSIS (1.5 hours)

### Step 2.1: Check for Missing Values

```python
# Check missing values
print("Missing Values:")
print(df.isnull().sum())
print(f"\nTotal missing: {df.isnull().sum().sum()}")
```

**Expected Result:** No missing values in this dataset!

### Step 2.2: Analyze Class Distribution (The Imbalance Problem)

```python
# SECTION 2: EXPLORATORY DATA ANALYSIS

# Check class distribution
print("Class Distribution:")
print(df['Class'].value_counts())
print("\nClass Proportion:")
print(df['Class'].value_counts(normalize=True))

# Visualize class imbalance
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Count plot
df['Class'].value_counts().plot(kind='bar', ax=axes[0], color=['green', 'red'])
axes[0].set_title('Fraud vs Legitimate Transactions (Count)')
axes[0].set_xlabel('Class (0=Legitimate, 1=Fraud)')
axes[0].set_ylabel('Number of Transactions')
axes[0].set_xticklabels(['Legitimate', 'Fraud'], rotation=0)

# Percentage plot
df['Class'].value_counts(normalize=True).plot(kind='pie', ax=axes[1], 
                                               labels=['Legitimate (99.83%)', 'Fraud (0.17%)'],
                                               autopct='%1.3f%%', colors=['green', 'red'])
axes[1].set_title('Fraud vs Legitimate Transactions (Percentage)')
axes[1].set_ylabel('')

plt.tight_layout()
plt.show()

print("\n⚠️ KEY INSIGHT: Highly imbalanced dataset!")
print("If we predict all transactions as legitimate, accuracy = 99.83%")
print("But we'd catch 0 frauds! This is why accuracy is misleading.")
```

### Step 2.3: Analyze Transaction Amounts

```python
# Compare amounts between legitimate and fraudulent transactions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Distribution of amounts
df[df['Class']==0]['Amount'].hist(bins=50, ax=axes[0], color='green', alpha=0.7, label='Legitimate')
df[df['Class']==1]['Amount'].hist(bins=50, ax=axes[0], color='red', alpha=0.7, label='Fraud')
axes[0].set_xlabel('Transaction Amount')
axes[0].set_ylabel('Frequency')
axes[0].set_title('Transaction Amount Distribution')
axes[0].legend()
axes[0].set_xlim([0, 1000])  # Zoom for better view

# Box plot comparison
data_to_plot = [df[df['Class']==0]['Amount'], df[df['Class']==1]['Amount']]
axes[1].boxplot(data_to_plot, labels=['Legitimate', 'Fraud'])
axes[1].set_ylabel('Amount')
axes[1].set_title('Amount Comparison: Fraud vs Legitimate')

plt.tight_layout()
plt.show()

print("Legitimate transactions statistics:")
print(df[df['Class']==0]['Amount'].describe())
print("\nFraud transactions statistics:")
print(df[df['Class']==1]['Amount'].describe())
```

### Step 2.4: Correlation Analysis

```python
# Create correlation matrix
correlation_matrix = df.corr()

# Plot correlation heatmap (for top features only)
plt.figure(figsize=(10, 8))
# Get correlations with fraud class
fraud_corr = correlation_matrix['Class'].sort_values(ascending=False)
top_features = fraud_corr.head(10).index.tolist()

sns.heatmap(df[top_features].corr(), annot=True, cmap='coolwarm', center=0)
plt.title('Correlation Matrix (Top Features with Fraud)')
plt.tight_layout()
plt.show()

print("Features most correlated with Fraud (positive correlation):")
print(fraud_corr[fraud_corr > 0].head(10))
```

### Step 2.5: Analyze V1-V28 Features

```python
# These are PCA-transformed features (privacy preserved)
# Check their distributions
v_features = [col for col in df.columns if col.startswith('V')]

# Compare distributions for fraud vs legitimate
fig, axes = plt.subplots(7, 4, figsize=(16, 14))
axes = axes.flatten()

for idx, feature in enumerate(v_features[:20]):
    df[df['Class']==0][feature].hist(bins=30, ax=axes[idx], alpha=0.6, 
                                      label='Legitimate', color='green')
    df[df['Class']==1][feature].hist(bins=30, ax=axes[idx], alpha=0.6, 
                                      label='Fraud', color='red')
    axes[idx].set_title(f'{feature} Distribution')
    if idx == 0:
        axes[idx].legend()

plt.tight_layout()
plt.show()

print("✓ EDA Complete! You've identified:")
print("  1. Class imbalance (0.17% fraud)")
print("  2. Fraud amounts differ from legitimate")
print("  3. Feature distributions differ between classes")
print("  4. Some features are more correlated with fraud")
```

---

## PHASE 3: DATA PREPROCESSING (1 hour)

### Step 3.1: Handle Class Imbalance

```python
# SECTION 3: DATA PREPROCESSING

# Check class distribution before preprocessing
print("Original Class Distribution:")
print(df['Class'].value_counts())

# Method 1: Using class weights (preferred for this problem)
# Method 2: Oversample minority class (SMOTE)
# We'll use Method 1 for simplicity, but mention Method 2

print("\nWe'll handle imbalance by:")
print("1. Using class_weight='balanced' in classifiers")
print("2. Using stratified train-test split")
print("3. Using appropriate evaluation metrics (not accuracy)")
```

### Step 3.2: Scale Features

```python
# Separate features and target
X = df.drop('Class', axis=1)
y = df['Class']

# Split into train and test sets BEFORE scaling (important!)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")
print(f"\nTraining set fraud ratio: {y_train.sum() / len(y_train):.4f}")
print(f"Test set fraud ratio: {y_test.sum() / len(y_test):.4f}")

# Scale features using StandardScaler
# Important: Fit scaler on TRAINING data only, then transform both
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\n✓ Features scaled successfully!")
print(f"Scaled training data shape: {X_train_scaled.shape}")
print(f"Mean of scaled features: {X_train_scaled.mean():.2e}")
print(f"Std of scaled features: {X_train_scaled.std():.2e}")

# Convert back to DataFrame for easier manipulation later
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X.columns)
```

**Why this order?**
- Split first: Prevents data leakage (test data influencing scaling)
- Scale training: Scaler learns from training data only
- Transform test: Apply same transformation to test data

---

## PHASE 4: MODEL DEVELOPMENT (2.5 hours)

### Step 4.1: Define Model Training Function

```python
# SECTION 4: MODEL DEVELOPMENT

# Create a dictionary to store results
model_results = {}

def train_evaluate_model(model, model_name, X_train, X_test, y_train, y_test):
    """
    Train a model and evaluate on test set
    Returns metrics dictionary
    """
    # Train the model
    model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of fraud
    
    # Calculate metrics
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    
    metrics = {
        'Model': model_name,
        'Accuracy': (tp + tn) / (tp + tn + fp + fn),
        'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'F1-Score': 2 * (tp / (tp + fp)) * (tp / (tp + fn)) / 
                   ((tp / (tp + fp)) + (tp / (tp + fn))) 
                   if (tp + fp) > 0 and (tp + fn) > 0 else 0,
        'ROC-AUC': roc_auc_score(y_test, y_pred_proba),
        'TP': tp,
        'FP': fp,
        'FN': fn,
        'TN': tn,
    }
    
    return model, metrics, y_pred, y_pred_proba
```

### Step 4.2: Train Logistic Regression

```python
# Model 1: Logistic Regression
print("=" * 60)
print("MODEL 1: LOGISTIC REGRESSION")
print("=" * 60)

lr_model, lr_metrics, lr_pred, lr_proba = train_evaluate_model(
    LogisticRegression(random_state=42, class_weight='balanced', max_iter=1000),
    'Logistic Regression',
    X_train_scaled, X_test_scaled, y_train, y_test
)

print(f"\nAccuracy: {lr_metrics['Accuracy']:.4f}")
print(f"Precision: {lr_metrics['Precision']:.4f}")
print(f"Recall: {lr_metrics['Recall']:.4f}")
print(f"F1-Score: {lr_metrics['F1-Score']:.4f}")
print(f"ROC-AUC: {lr_metrics['ROC-AUC']:.4f}")

model_results['Logistic Regression'] = lr_metrics
```

### Step 4.3: Train Decision Tree

```python
# Model 2: Decision Tree
print("\n" + "=" * 60)
print("MODEL 2: DECISION TREE")
print("=" * 60)

dt_model, dt_metrics, dt_pred, dt_proba = train_evaluate_model(
    DecisionTreeClassifier(random_state=42, class_weight='balanced', max_depth=10),
    'Decision Tree',
    X_train_scaled, X_test_scaled, y_train, y_test
)

print(f"\nAccuracy: {dt_metrics['Accuracy']:.4f}")
print(f"Precision: {dt_metrics['Precision']:.4f}")
print(f"Recall: {dt_metrics['Recall']:.4f}")
print(f"F1-Score: {dt_metrics['F1-Score']:.4f}")
print(f"ROC-AUC: {dt_metrics['ROC-AUC']:.4f}")

model_results['Decision Tree'] = dt_metrics
```

### Step 4.4: Train Random Forest

```python
# Model 3: Random Forest
print("\n" + "=" * 60)
print("MODEL 3: RANDOM FOREST")
print("=" * 60)

rf_model, rf_metrics, rf_pred, rf_proba = train_evaluate_model(
    RandomForestClassifier(n_estimators=100, random_state=42, 
                          class_weight='balanced', max_depth=15, n_jobs=-1),
    'Random Forest',
    X_train_scaled, X_test_scaled, y_train, y_test
)

print(f"\nAccuracy: {rf_metrics['Accuracy']:.4f}")
print(f"Precision: {rf_metrics['Precision']:.4f}")
print(f"Recall: {rf_metrics['Recall']:.4f}")
print(f"F1-Score: {rf_metrics['F1-Score']:.4f}")
print(f"ROC-AUC: {rf_metrics['ROC-AUC']:.4f}")

model_results['Random Forest'] = rf_metrics
```

### Step 4.5: Train Support Vector Machine

```python
# Model 4: Support Vector Machine
print("\n" + "=" * 60)
print("MODEL 4: SUPPORT VECTOR MACHINE")
print("=" * 60)

svm_model, svm_metrics, svm_pred, svm_proba = train_evaluate_model(
    SVC(kernel='rbf', probability=True, random_state=42, class_weight='balanced'),
    'Support Vector Machine',
    X_train_scaled, X_test_scaled, y_train, y_test
)

print(f"\nAccuracy: {svm_metrics['Accuracy']:.4f}")
print(f"Precision: {svm_metrics['Precision']:.4f}")
print(f"Recall: {svm_metrics['Recall']:.4f}")
print(f"F1-Score: {svm_metrics['F1-Score']:.4f}")
print(f"ROC-AUC: {svm_metrics['ROC-AUC']:.4f}")

model_results['Support Vector Machine'] = svm_metrics
```

---

## PHASE 5: MODEL EVALUATION & COMPARISON (1.5 hours)

### Step 5.1: Create Comparison Table

```python
# SECTION 5: MODEL EVALUATION

# Create results DataFrame
results_df = pd.DataFrame(model_results).T
results_df = results_df[['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']]

print("=" * 80)
print("MODEL COMPARISON TABLE")
print("=" * 80)
print(results_df.round(4))

# Rank models by each metric
print("\n" + "=" * 80)
print("MODEL RANKING BY KEY METRICS")
print("=" * 80)
for metric in ['Recall', 'Precision', 'F1-Score', 'ROC-AUC']:
    ranked = results_df[metric].sort_values(ascending=False)
    print(f"\n{metric}:")
    for i, (model, score) in enumerate(ranked.items(), 1):
        print(f"  {i}. {model}: {score:.4f}")
```

### Step 5.2: Confusion Matrix Visualization

```python
# Visualize confusion matrices for all models
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

models_to_plot = [
    (lr_pred, 'Logistic Regression', axes[0]),
    (dt_pred, 'Decision Tree', axes[1]),
    (rf_pred, 'Random Forest', axes[2]),
    (svm_pred, 'Support Vector Machine', axes[3])
]

for predictions, model_name, ax in models_to_plot:
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Legitimate', 'Fraud'])
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    ax.set_title(f'{model_name} Confusion Matrix')

plt.tight_layout()
plt.show()
```

### Step 5.3: ROC Curves Comparison

```python
# Plot ROC curves for all models
plt.figure(figsize=(10, 8))

fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_proba)
fpr_dt, tpr_dt, _ = roc_curve(y_test, dt_proba)
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_proba)
fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_proba)

plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC={lr_metrics["ROC-AUC"]:.4f})', linewidth=2)
plt.plot(fpr_dt, tpr_dt, label=f'Decision Tree (AUC={dt_metrics["ROC-AUC"]:.4f})', linewidth=2)
plt.plot(fpr_rf, tpr_rf, label=f'Random Forest (AUC={rf_metrics["ROC-AUC"]:.4f})', linewidth=2)
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC={svm_metrics["ROC-AUC"]:.4f})', linewidth=2)

# Plot random classifier line
plt.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Random Classifier')

plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Model Comparison', fontsize=14, fontweight='bold')
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

print("✓ ROC-AUC measures the probability that the model ranks a random fraud")
print("  higher than a random legitimate transaction")
print("✓ Higher AUC = Better model (1.0 is perfect, 0.5 is random)")
```

### Step 5.4: Detailed Classification Report

```python
# Get detailed classification report for best model
print("=" * 80)
print("DETAILED CLASSIFICATION REPORT - BEST MODEL")
print("=" * 80)

best_model = rf_model  # Assuming Random Forest is best (you'll verify)
y_pred_best = best_model.predict(X_test_scaled)

print(classification_report(y_test, y_pred_best, target_names=['Legitimate', 'Fraud']))

print("\nMetric Explanations:")
print("- Precision: Of predicted frauds, how many were actually fraud?")
print("  High precision = Few false positives (good for customer satisfaction)")
print("- Recall: Of actual frauds, how many did we catch?")
print("  High recall = Few false negatives (good for fraud detection)")
print("- F1-Score: Harmonic mean of precision and recall")
print("- Support: Number of samples in each class")
```

---

## PHASE 6: FEATURE IMPORTANCE & HYPERPARAMETER TUNING (1.5 hours)

### Step 6.1: Feature Importance (Tree-based Models)

```python
# SECTION 6: FEATURE IMPORTANCE & INTERPRETABILITY

# Random Forest Feature Importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': rf_model.feature_importances_
}).sort_values('Importance', ascending=False)

print("Top 15 Features - Random Forest:")
print(feature_importance.head(15))

# Visualize top features
plt.figure(figsize=(10, 8))
top_n = 15
top_features = feature_importance.head(top_n)

plt.barh(range(top_n), top_features['Importance'].values, color='steelblue')
plt.yticks(range(top_n), top_features['Feature'].values)
plt.xlabel('Feature Importance', fontsize=12)
plt.title(f'Top {top_n} Features - Random Forest', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()

for i, v in enumerate(top_features['Importance'].values):
    plt.text(v + 0.001, i, f'{v:.4f}', va='center')

plt.tight_layout()
plt.show()

print("\n✓ KEY INSIGHT:")
print("These features have the most impact on fraud detection")
print("V4, V12, V14 are the strongest fraud predictors")
```

### Step 6.2: Hyperparameter Tuning with GridSearchCV

```python
# Tune Random Forest hyperparameters
print("\n" + "=" * 80)
print("HYPERPARAMETER TUNING - RANDOM FOREST")
print("=" * 80)

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

print("Starting GridSearchCV... this may take a few minutes")

# Use F1-score as scoring metric (better than accuracy for imbalanced data)
grid_search = GridSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced', n_jobs=-1),
    param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='f1',
    n_jobs=-1,
    verbose=1
)

grid_search.fit(X_train_scaled, y_train)

print(f"\nBest Parameters: {grid_search.best_params_}")
print(f"Best F1-Score (CV): {grid_search.best_score_:.4f}")

# Evaluate tuned model
y_pred_tuned = grid_search.predict(X_test_scaled)
y_proba_tuned = grid_search.predict_proba(X_test_scaled)[:, 1]

print(f"\nTest Set Performance:")
print(f"Precision: {precision_score(y_test, y_pred_tuned):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_tuned):.4f}")
print(f"F1-Score: {f1_score(y_test, y_pred_tuned):.4f}")
print(f"ROC-AUC: {roc_auc_score(y_test, y_proba_tuned):.4f}")

best_model = grid_search.best_estimator_
```

---

## PHASE 7: LIME EXPLAINABILITY (1 hour)

### Step 7.1: Install and Setup LIME

```python
# Install LIME
# !pip install lime

import lime
import lime.lime_tabular

# Create LIME explainer
explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train_scaled.values,
    feature_names=X.columns,
    class_names=['Legitimate', 'Fraud'],
    mode='classification',
    random_state=42
)
```

### Step 7.2: Explain Individual Predictions

```python
# SECTION 7: LIME EXPLAINABILITY

print("=" * 80)
print("EXPLAINING INDIVIDUAL PREDICTIONS")
print("=" * 80)

# Find examples of fraudulent and legitimate transactions
fraud_indices = y_test[y_test == 1].index
legit_indices = y_test[y_test == 0].index

# Get a fraudulent transaction
fraud_example_idx = X_test.index[fraud_indices[0]]
fraud_example = X_test_scaled.loc[fraud_example_idx].values.reshape(1, -1)

print("\n" + "-" * 80)
print("EXAMPLE 1: FRAUDULENT TRANSACTION EXPLANATION")
print("-" * 80)

fraud_explanation = explainer.explain_instance(
    fraud_example[0],
    best_model.predict_proba,
    num_features=10
)

print("Explanation for predicted fraud transaction:")
fraud_explanation.show_in_notebook()

# Get the explanation components
print("\nTop features contributing to FRAUD prediction:")
fraud_exp_list = fraud_explanation.as_list()
for feature, contribution in fraud_exp_list[:5]:
    print(f"  {feature}: {contribution:.4f}")

# Get a legitimate transaction
legit_example_idx = X_test.index[legit_indices[100]]
legit_example = X_test_scaled.loc[legit_example_idx].values.reshape(1, -1)

print("\n" + "-" * 80)
print("EXAMPLE 2: LEGITIMATE TRANSACTION EXPLANATION")
print("-" * 80)

legit_explanation = explainer.explain_instance(
    legit_example[0],
    best_model.predict_proba,
    num_features=10
)

print("Explanation for predicted legitimate transaction:")
legit_explanation.show_in_notebook()

print("\nTop features contributing to LEGITIMATE prediction:")
legit_exp_list = legit_explanation.as_list()
for feature, contribution in legit_exp_list[:5]:
    print(f"  {feature}: {contribution:.4f}")
```

### Step 7.3: Analyze Multiple Examples

```python
# Analyze multiple fraud examples to find patterns
print("\n" + "=" * 80)
print("PATTERN ANALYSIS: Multiple Fraud Examples")
print("=" * 80)

fraud_features_count = {}

for i in range(5):  # Analyze 5 fraud examples
    fraud_idx = X_test.index[fraud_indices[i]]
    example = X_test_scaled.loc[fraud_idx].values.reshape(1, -1)
    
    explanation = explainer.explain_instance(
        example[0],
        best_model.predict_proba,
        num_features=5
    )
    
    print(f"\nFraud Transaction {i+1}:")
    for feature, contribution in explanation.as_list()[:3]:
        feature_name = feature.split()[0]
        fraud_features_count[feature_name] = fraud_features_count.get(feature_name, 0) + 1
        print(f"  {feature}")

print("\n" + "=" * 80)
print("MOST COMMON FRAUD INDICATORS:")
print("=" * 80)
for feature, count in sorted(fraud_features_count.items(), key=lambda x: x[1], reverse=True):
    print(f"  {feature}: appears in {count}/5 fraud examples")
```

---

## PHASE 8: ANALYSIS & CONCLUSIONS (1 hour)

### Step 8.1: Key Findings Summary

```python
# SECTION 8: ANALYSIS & CONCLUSIONS

print("=" * 80)
print("KEY FINDINGS & INSIGHTS")
print("=" * 80)

print("""
1. CLASS IMBALANCE:
   - Fraud represents only 0.17% of transactions
   - Requires special handling: class weighting, proper metrics, stratified splits
   - Accuracy alone is misleading

2. BEST MODEL PERFORMANCE:
""")

best_metrics = results_df.loc[results_df['F1-Score'].idxmax()]
print(f"""
   Model: {results_df['F1-Score'].idxmax()}
   - ROC-AUC: {best_metrics['ROC-AUC']:.4f} (out of 1.0)
   - Precision: {best_metrics['Precision']:.4f} (fewer false alarms)
   - Recall: {best_metrics['Recall']:.4f} (catches more frauds)
   - F1-Score: {best_metrics['F1-Score']:.4f}

3. TOP FRAUD INDICATORS:
   - Primary: V4, V12, V14 (PCA-transformed features)
   - Secondary: V3, V10, V27
   - Amount is less important than feature values

4. FALSE POSITIVES vs FALSE NEGATIVES:
   - We optimize for Recall (catching frauds)
   - Accept some false positives (inconvenience customers)
   - Better to flag legitimate as fraud than miss actual fraud

5. DEPLOYMENT RECOMMENDATIONS:
   - Set fraud probability threshold for customer friction tolerance
   - Monitor model performance monthly
   - Retrain quarterly with new fraud patterns
   - Use ensemble of models for production
""")
```

### Step 8.2: Save Models for Deployment

```python
import joblib

# Save the best model
joblib.dump(best_model, 'fraud_detection_model.pkl')

# Save the scaler (crucial for preprocessing new data)
joblib.dump(scaler, 'feature_scaler.pkl')

print("✓ Model saved as 'fraud_detection_model.pkl'")
print("✓ Scaler saved as 'feature_scaler.pkl'")

print("\nFor production deployment:")
print("""
# Load and use
model = joblib.load('fraud_detection_model.pkl')
scaler = joblib.load('feature_scaler.pkl')

# Predict on new transaction
new_transaction = [...]  # 30 features
new_transaction_scaled = scaler.transform([new_transaction])
prediction = model.predict(new_transaction_scaled)
probability = model.predict_proba(new_transaction_scaled)[0, 1]

if probability > 0.5:
    print(f"FRAUD ALERT! Confidence: {probability:.2%}")
""")
```

---

## Common Issues & Solutions

| Issue | Solution |
|-------|----------|
| Low recall (missing frauds) | Increase threshold / adjust class weights |
| Low precision (many false alarms) | Increase threshold / tune hyperparameters |
| Model overfitting | Reduce tree depth / add regularization |
| Features not scaled | Always use StandardScaler before training |
| Data leakage | Split BEFORE scaling, fit scaler on TRAIN only |
| Imbalanced results in train/test | Use stratify=y in train_test_split |

---

## Checkpoints

- ✅ **Checkpoint 1 (2 hours):** Data exploration, EDA visualizations complete
- ✅ **Checkpoint 2 (4 hours):** 2+ models trained and compared
- ✅ **Checkpoint 3 (6 hours):** All 4 models trained, evaluation metrics calculated
- ✅ **Checkpoint 4 (8 hours):** Feature importance extracted, LIME explanations provided

---

## Final Checklist

Before submitting:

- [ ] All 4+ models implemented
- [ ] Train-test split used correctly (no data leakage)
- [ ] Class imbalance handled explicitly
- [ ] Confusion matrices visualized for all models
- [ ] ROC curves plotted and compared
- [ ] Model evaluation metrics calculated (Precision, Recall, F1, ROC-AUC)
- [ ] Hyperparameter tuning completed with GridSearchCV
- [ ] Feature importance extracted and visualized
- [ ] LIME explanations provided (3-5 examples)
- [ ] Code is well-commented
- [ ] Results documented in a report
- [ ] Models saved for deployment
- [ ] Clear insights and conclusions written

**Good luck! 🚀**
