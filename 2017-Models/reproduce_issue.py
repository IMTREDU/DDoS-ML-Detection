import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_recall_curve

# Simulate binary classification
y_test = np.array([0, 1, 0, 1, 0, 1])
classes = np.unique(y_test)
y_test_bin = label_binarize(y_test, classes=classes)

print(f"y_test_bin shape: {y_test_bin.shape}")

# Simulate predict_proba output (2 columns for binary)
y_score = np.array([
    [0.9, 0.1],
    [0.2, 0.8],
    [0.8, 0.2],
    [0.3, 0.7],
    [0.95, 0.05],
    [0.4, 0.6]
])

print(f"y_score shape: {y_score.shape}")

# The problematic loop from dt.PY
try:
    for i, class_id in enumerate(range(y_score.shape[1])):
        print(f"Processing class {i}...")
        # This should fail for i=1 if y_test_bin has 1 column
        precision, recall, _ = precision_recall_curve(y_test_bin[:, i], y_score[:, i])
        print(f"Class {i} success")
except Exception as e:
    print(f"Caught expected error: {e}")
