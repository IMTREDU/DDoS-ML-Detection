import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# ==========================================
# TRAIN ON CICDDoS2019 (KNN)
# ==========================================
print("\n--- Training on CICDDoS2019 (KNN) ---")

# 1. Load Dataset
file_path_2019 = os.path.join(os.path.dirname(__file__), '..', 'DataSets', 'cicddos2019.csv')
if not os.path.exists(file_path_2019):
    raise FileNotFoundError(f"Could not find {file_path_2019}")

df = pd.read_csv(file_path_2019)
print(f"Original Dataset Shape: {df.shape}")

# 2. Basic Cleaning
df.columns = df.columns.str.strip()
df.drop(columns=['Timestamp'], errors='ignore', inplace=True)
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

# Drop non-numeric columns except Label
non_numeric_cols = df.select_dtypes(include=['object']).columns.drop('Label', errors='ignore')
df.drop(columns=non_numeric_cols, inplace=True)

# Remove rare classes (<10 samples)
min_samples = 10
class_counts = df['Label'].value_counts()
valid_classes = class_counts[class_counts >= min_samples].index
df = df[df['Label'].isin(valid_classes)]
print(f"Dataset Shape After Cleaning and Dropping Rare Classes: {df.shape}")

# 3. Label Encoding
le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])
print("Labels encoded:")
for k, v in zip(le.classes_, le.transform(le.classes_)):
    print(f"  {k:20s} -> {v}")

# 4. Train / Test Split
X = df.drop('Label', axis=1)
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
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
# 6. Hyperparameter Tuning (optional)
# ==========================================
param_grid = {
    'knn__estimator__n_neighbors': [3, 5, 7, 9],
    'knn__estimator__weights': ['uniform', 'distance'],
    'knn__estimator__metric': ['euclidean', 'manhattan']
}

grid_search = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=3,            # smaller CV to save time
    scoring='accuracy',
    n_jobs=-1,
    verbose=2
)

print("\nRunning GridSearchCV for KNN hyperparameter tuning...")
grid_search.fit(X_train, y_train)
best_pipeline = grid_search.best_estimator_
print("Best Hyperparameters:", grid_search.best_params_)

# ==========================================
# 7. Train & Predict
# ==========================================
y_pred = best_pipeline.predict(X_test)

# 8. Overfitting Check
train_acc = best_pipeline.score(X_train, y_train)
test_acc = accuracy_score(y_test, y_pred)
print(f"\nTraining Accuracy: {train_acc:.4f}")
print(f"Testing Accuracy:  {test_acc:.4f}")
print(f"Difference:        {train_acc - test_acc:.4f}")

if abs(train_acc - test_acc) > 0.05:
    print("WARNING: Possible overfitting detected.")
else:
    print("GOOD: No significant overfitting detected.")

# 9. Cross-Validation
print("\nRunning 5-Fold Cross-Validation on best pipeline...")
cv_scores = cross_val_score(best_pipeline, X_train, y_train, cv=5, scoring='accuracy')
print(f"CV Scores: {cv_scores}")
print(f"Average CV Accuracy: {cv_scores.mean():.4f}")

# 10. Classification Report
print("\nTest Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# 11. Confusion Matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=False, cmap='Blues', cbar=True)
plt.title('Confusion Matrix - KNN (CICDDoS2019)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.show()

# 12. Multiclass Precision-Recall Curves
print("\nPlotting Multiclass Precision-Recall Curves...")
classes = np.unique(y_test)
y_test_bin = label_binarize(y_test, classes=classes)
y_score = best_pipeline.predict_proba(X_test)

plt.figure(figsize=(10, 8))
for i, class_id in enumerate(classes):
    precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
    class_name = le.inverse_transform([class_id])[0]
    plt.plot(recall, precision, label=class_name)

plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Multiclass Precision-Recall Curves (CICDDoS2019)")
plt.legend(fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.show()