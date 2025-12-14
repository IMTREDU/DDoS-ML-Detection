import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_curve
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# ==========================================
#           TRAIN ON CICIDS2017
# ==========================================
print("--- Training on CICIDS2017 (Logistic Regression) ---")

# 1. Load Dataset
file_path_2017 = os.path.join(
    os.path.dirname(__file__), '..', 'DataSets', 'cicids2017.csv'
)

if not os.path.exists(file_path_2017):
    raise FileNotFoundError(f"Could not find {file_path_2017}")

df = pd.read_csv(file_path_2017)
print(f"Original Dataset Shape: {df.shape}")

# 2. Basic Cleaning
df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# 3. Label Encoding
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])
print(f"Labels encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# 4. Train / Test Split
X = df.drop('Label', axis=1)
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# ==========================================
# 5. Pipeline (NO DATA LEAKAGE)
# ==========================================
pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('pca', PCA(n_components=0.95, random_state=42)),
    ('logreg', LogisticRegression(
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    ))
])

# ==========================================
# 6. Model Training
# ==========================================
print("[2017] Training Logistic Regression pipeline...")
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# ==========================================
# OVERFITTING CHECK
# ==========================================
print("\n--- Overfitting Check ---")

train_accuracy = pipeline.score(X_train, y_train)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy:  {test_accuracy:.4f}")
print(f"Difference:        {train_accuracy - test_accuracy:.4f}")

if train_accuracy - test_accuracy > 0.05:
    print("WARNING: Possible overfitting detected.")
else:
    print("GOOD: No significant overfitting detected.")

# ==========================================
# 7. Proper Cross-Validation
# ==========================================
print("\nRunning 5-Fold Cross-Validation (Leakage-Free)...")

cv_scores = cross_val_score(
    pipeline,
    X_train,
    y_train,
    cv=5,
    scoring='accuracy'
)

print(f"CV Scores: {cv_scores}")
print(f"Average CV Accuracy: {cv_scores.mean():.4f}")

# ==========================================
# 8. Evaluation Metrics
# ==========================================
print("\n[2017 Test Classification Report]")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Logistic Regression (2017)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Precision-Recall Curve
print("[2017] Plotting Precision-Recall Curve...")
y_probs = pipeline.predict_proba(X_test)[:, 1]
precision, recall, _ = precision_recall_curve(y_test, y_probs)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, label='Logistic Regression')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (CICIDS2017)')
plt.legend()
plt.grid(True)
plt.show()