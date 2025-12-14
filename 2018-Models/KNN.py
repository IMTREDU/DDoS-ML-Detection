import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_curve
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# ==========================================
#           TRAIN ON CICIDS2018 (KNN)
# ==========================================
print("--- Training on CICIDS2018 (KNN) ---")

# 1. Load Dataset
file_path_2018 = os.path.join(
    os.path.dirname(__file__), '..', 'DataSets', 'cicids2018.csv'
)

if not os.path.exists(file_path_2018):
    raise FileNotFoundError(f"Could not find {file_path_2018}")

df = pd.read_csv(file_path_2018)
print(f"Original Dataset Shape: {df.shape}")

# 2. Basic Cleaning
df.columns = df.columns.str.strip()
df.drop(columns=['Timestamp'], inplace=True, errors='ignore')
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
# 5. Pipeline (Scaling, SMOTE, PCA, KNN)
# ==========================================
knn_model = OneVsRestClassifier(KNeighborsClassifier(n_jobs=-1))

pipeline = Pipeline(steps=[
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('pca', PCA(n_components=0.95, random_state=42)),
    ('knn', knn_model)
])

# ==========================================
# 6. Hyperparameter Tuning (GridSearchCV)
# ==========================================
param_grid = {
    'knn__estimator__n_neighbors': [3, 5, 7, 9],
    'knn__estimator__weights': ['uniform', 'distance'],
    'knn__estimator__metric': ['euclidean', 'manhattan']
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

print("[2018] Running GridSearchCV for KNN hyperparameter tuning...")
grid_search.fit(X_train, y_train)

# Best pipeline after tuning
best_pipeline = grid_search.best_estimator_
print("Best Hyperparameters found:")
print(grid_search.best_params_)

# ==========================================
# 7. Predictions and Accuracy Check
# ==========================================
y_pred = best_pipeline.predict(X_test)

train_accuracy = best_pipeline.score(X_train, y_train)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"\nTraining Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy:  {test_accuracy:.4f}")
print(f"Difference:        {train_accuracy - test_accuracy:.4f}")

if train_accuracy - test_accuracy > 0.05:
    print("WARNING: Possible overfitting detected.")
else:
    print("GOOD: No significant overfitting detected.")

# ==========================================
# 8. Proper Cross-Validation
# ==========================================
print("\nRunning 5-Fold Cross-Validation on best pipeline...")
cv_scores = cross_val_score(
    best_pipeline,
    X_train,
    y_train,
    cv=5,
    scoring='accuracy'
)
print(f"CV Scores: {cv_scores}")
print(f"Average CV Accuracy: {cv_scores.mean():.4f}")

# ==========================================
# 9. Evaluation Metrics
# ==========================================
print("\n[2018 Test Classification Report]")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - KNN (2018)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# Precision-Recall Curve (multiclass)
print("[2018] Plotting Precision-Recall Curve...")
y_test_bin = label_binarize(y_test, classes=np.arange(len(le.classes_)))
y_score = best_pipeline.predict_proba(X_test)

for i, class_name in enumerate(le.classes_):
    precision, recall, _ = precision_recall_curve(
        y_test_bin[:, i],
        y_score[:, i]
    )
    plt.plot(recall, precision, label=class_name)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Multiclass Precision-Recall Curves (CICIDS2018)")
plt.legend()
plt.grid(True)
plt.show()
