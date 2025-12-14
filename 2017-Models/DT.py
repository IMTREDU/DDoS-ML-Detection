import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, label_binarize
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# ==========================================
#           TRAIN ON CICIDS2017
# ==========================================
print("--- Training on CICIDS2017 (Decision Tree) ---")

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
# 5. Pipeline (SMOTE + Decision Tree)
# ==========================================
pipeline = Pipeline(steps=[
    ('smote', SMOTE(random_state=42)),
    ('dt', DecisionTreeClassifier(
        random_state=42,
        max_depth=None,      # Can tune for pruning
        min_samples_split=2
    ))
])

# ==========================================
# 6. Model Training
# ==========================================
print("[2017] Training Decision Tree pipeline...")
pipeline.fit(X_train, y_train)

# Predictions
y_pred = pipeline.predict(X_test)

# ==========================================
# 7. Overfitting Check
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
# 8. Cross-Validation
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
# 9. Classification Report
# ==========================================
print("\n[2017 Test Classification Report]")
print(classification_report(y_test, y_pred))

# ==========================================
# 10. Confusion Matrix
# ==========================================
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - Decision Tree (CICIDS2017)')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# ==========================================
# 11. Decision Tree Visualization
# ==========================================
dt_model = pipeline.named_steps['dt']
plt.figure(figsize=(20, 10))
plot_tree(
    dt_model,
    feature_names=X.columns,
    class_names=le.classes_,
    filled=True,
    rounded=True,
    fontsize=10
)
plt.title("Decision Tree Visualization (CICIDS2017)")
plt.show()

# ==========================================
# 12. Precision-Recall Curves
# ==========================================
print("[2017] Plotting Precision-Recall Curve...")
y_test_bin = label_binarize(y_test, classes=np.arange(len(le.classes_)))

if hasattr(dt_model, "predict_proba"):
    y_score = pipeline.predict_proba(X_test)

    # Fix for binary classification: ensure 2 columns if label_binarize returns 1
    if y_test_bin.shape[1] == 1 and len(le.classes_) == 2:
        y_test_bin = np.hstack([1 - y_test_bin, y_test_bin])

    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(le.classes_):
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        plt.plot(recall, precision, label=class_name)

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Multiclass Precision-Recall Curves (CICIDS2017)")
    plt.legend()
    plt.grid(True)
    plt.show()
else:
    print("Decision Tree does not support probability predictions for Precision-Recall curves unless using 'predict_proba'.")