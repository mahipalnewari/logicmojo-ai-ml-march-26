# Credit Card Fraud Detection - One-Page Quick Reference

## 🎯 PROJECT IN 60 SECONDS

**Problem:** Detect fraudulent credit card transactions (0.17% of 284K) AND explain why  
**Data:** 31 features, 284,807 transactions, 492 frauds  
**Challenge:** Class imbalance 577:1  
**Solution:** Train 4 ML models, tune hyperparameters, explain with LIME  

---

## 🚀 QUICK START

```bash
# 1. Download dataset
https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# 2. Install libraries
pip install pandas numpy scikit-learn matplotlib seaborn lime

# 3. Follow PROJECT_INSTRUCTIONS.md step-by-step

# 4. Reference SOLUTION_DESCRIPTION.md for explanations
```

---

## 📋 8-HOUR PROJECT TIMELINE

| Time | Phase | What to Do | Output |
|------|-------|-----------|--------|
| 08:00-09:30 | EDA | Load data, visualize classes, analyze features | Charts showing fraud patterns |
| 09:30-11:00 | Preprocess | Scale features, stratified split, handle imbalance | Train & test sets, scaler |
| 11:00-12:30 | Models | Implement 4 algorithms | 4 trained models |
| 12:30-13:30 | LUNCH | ☕ Break | |
| 13:30-15:00 | Evaluate | Compare models, confusion matrix, ROC curves | Metrics table, visualizations |
| 15:00-16:30 | Tuning & Importance | GridSearchCV, feature importance | Best model, feature ranking |
| 16:30-17:00 | LIME + Docs | Explain predictions, write summary | Examples, recommendations |

---

## 💻 CRITICAL CODE PATTERNS

### Pattern 1: Proper Train-Test-Scale
```python
# ✅ CORRECT ORDER
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y  # Stratify!
)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)  # Fit on TRAIN
X_test_scaled = scaler.transform(X_test)        # Transform TEST
```

### Pattern 2: Model Training
```python
model = LogisticRegression(class_weight='balanced')  # Handle imbalance!
model.fit(X_train_scaled, y_train)
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]  # For ROC-AUC
```

### Pattern 3: Evaluation
```python
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba)  # Use probabilities!

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
```

### Pattern 4: Feature Importance
```python
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values('Importance', ascending=False)
```

### Pattern 5: LIME Explanation
```python
import lime.lime_tabular

explainer = lime.lime_tabular.LimeTabularExplainer(
    X_train_scaled.values, feature_names=X.columns.tolist()
)
explanation = explainer.explain_instance(
    transaction_features, model.predict_proba, num_features=10
)
print(explanation.as_list())  # [(feature, weight), ...]
```

---

## 📊 KEY METRICS (NOT ACCURACY!)

| Metric | Formula | Meaning | For This Problem |
|--------|---------|---------|------------------|
| **Precision** | TP/(TP+FP) | Of flagged frauds, how many correct? | Want: HIGH (few false alarms) |
| **Recall** | TP/(TP+FN) | Of actual frauds, how many caught? | Want: HIGH (catch frauds) |
| **F1-Score** | 2×(P×R)/(P+R) | Harmonic mean, balanced | Use for model comparison |
| **ROC-AUC** | Area under curve | Ranking ability of model | Want: >0.95 (excellent) |
| **Accuracy** | (TP+TN)/Total | % correct | ❌ MISLEADING (99.83% by predicting all legit!) |

---

## 4️⃣ ALGORITHMS TO IMPLEMENT

| Algorithm | Why | Hyperparameters | Pros | Cons |
|-----------|-----|-----------------|------|------|
| **Logistic Regression** | Baseline linear model | C, max_iter, solver | Fast, interpretable | Limited power |
| **Decision Tree** | Single tree splits | max_depth, min_samples_split | Interpretable rules | Overfits |
| **Random Forest** | Ensemble of 100 trees | n_estimators, max_depth | Robust, feature importance | Less interpretable |
| **SVM** | Find optimal hyperplane | kernel, C, gamma | High accuracy, high-dim | Slow, needs scaling |

---

## 🔑 HYPERPARAMETER TUNING

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [10, 15, 20],
    'min_samples_split': [5, 10],
    'min_samples_leaf': [2, 4]
}

grid = GridSearchCV(
    RandomForestClassifier(class_weight='balanced'),
    param_grid, cv=5,          # 5-fold CV
    scoring='f1',              # Use F1, not accuracy!
    n_jobs=-1
)
grid.fit(X_train_scaled, y_train)

print(grid.best_params_)       # Best configuration
print(grid.best_score_)        # CV score
predictions = grid.predict(X_test_scaled)  # Use best model
```

---

## ⚠️ TOP 5 MISTAKES

| ❌ Wrong | ✅ Correct |
|---------|-----------|
| Scale entire dataset before split | Split FIRST, scale train, transform test |
| Use accuracy as metric | Use Precision, Recall, F1-Score, ROC-AUC |
| Ignore class imbalance | Use class_weight='balanced' or SMOTE |
| Tune on test set | Use GridSearchCV with cross-validation |
| One model is enough | Train 4+ models and compare |

---

## 📈 EXPECTED RESULTS

```
EXCELLENT MODEL:
  Precision: 0.90    (90% of flagged are fraud)
  Recall:    0.85    (catch 85% of actual fraud)
  F1-Score:  0.87    (good balance)
  ROC-AUC:   0.96    (excellent ranking)

CONFUSION MATRIX:
  True Negatives:  56,600   (legit approved) ✓
  False Positives: 300      (legit blocked) - tolerable
  False Negatives: 80       (fraud missed) - minimized
  True Positives:  570      (fraud caught) ✓
```

---

## 📁 DELIVERABLES CHECKLIST

- [ ] **fraud_detection.ipynb** - Complete notebook with all 10 sections
- [ ] **Analysis Report** - 1-2 pages summarizing findings
- [ ] **Visualizations** - Saved figures (confusion matrix, ROC, features)
- [ ] **Model Files** - fraud_detection_model.pkl + feature_scaler.pkl
- [ ] **Explanations** - 5+ LIME examples with interpretations
- [ ] **Code Comments** - Every section explained

---

## 🎯 EVALUATION RUBRIC (100 points)

| Category | Points | Checklist |
|----------|--------|-----------|
| **EDA** | 15 | Class imbalance visualized, feature distributions shown |
| **Preprocessing** | 15 | Proper scaling, stratified split, class handling |
| **Models** | 20 | 4+ algorithms trained, predictions made |
| **Evaluation** | 20 | Confusion matrices, precision/recall/F1/ROC-AUC |
| **Tuning & Importance** | 15 | GridSearchCV used, top features identified |
| **Code Quality** | 10 | Well-organized, comments, no data leakage |
| **Report** | 5 | Findings documented, recommendations given |

---

## 🚨 DEBUG CHECKLIST

**Problem: Low fraud recall (missing frauds)**
- ✓ Check if class imbalance is handled (class_weight='balanced'?)
- ✓ Lower prediction threshold (from 0.5 to 0.3)
- ✓ Increase max_depth for tree models
- ✓ Try different algorithms

**Problem: High false positive rate (blocking too many legit)**
- ✓ Increase prediction threshold (from 0.5 to 0.7)
- ✓ Reduce max_depth
- ✓ Increase min_samples_leaf

**Problem: Model training takes forever**
- ✓ Subsample data: `df.sample(frac=0.5)`
- ✓ Reduce hyperparameter search space
- ✓ Use fewer CV folds: `cv=3`
- ✓ Try with Random Forest first

---

## 📚 REFERENCE DOCUMENTS IN PACKAGE

1. **README.md** - Project overview & getting started
2. **PROJECT_BRIEF.md** - Complete requirements
3. **PROJECT_INSTRUCTIONS.md** - Step-by-step guide with code
4. **fraud_detection_solution.ipynb** - Full working solution
5. **SOLUTION_DESCRIPTION.md** - Deep explanations of techniques

---

## 🎓 LEARNING OUTCOMES

After this project, you'll understand:
✅ Imbalanced classification problems  
✅ Model evaluation beyond accuracy  
✅ Feature importance extraction  
✅ Individual prediction explanations (LIME)  
✅ Hyperparameter tuning best practices  
✅ Production ML considerations  

---

**READY TO START?**

1. Download dataset (5 min)
2. Read PROJECT_BRIEF.md (15 min)  
3. Follow PROJECT_INSTRUCTIONS.md (6-8 hours)
4. Reference SOLUTION_DESCRIPTION.md (anytime)
5. Check solution.ipynb (for patterns)

**Good luck! 🚀**
