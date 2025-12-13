import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_recall_curve

# ==========================================
# PHASE 1: TRAIN ON CICIDS2017
# ==========================================
print("--- PHASE 1: Training on CICIDS2017 ---")

file_path_2017 = os.path.join(os.path.dirname(__file__), 'cicids2017.csv')
if not os.path.exists(file_path_2017):
    raise FileNotFoundError(f"Could not find {file_path_2017}")

df = pd.read_csv(file_path_2017)

df.columns = df.columns.str.strip()
df.replace([np.inf, -np.inf], np.nan, inplace=True)
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

le = LabelEncoder()
df['Label'] = le.fit_transform(df['Label'])
print(f"[2017] Labels encoded: {dict(zip(le.classes_, le.transform(le.classes_)))}")

X = df.drop('Label', axis=1)
y = df['Label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("[2017] Applying SMOTE...")
smote = SMOTE(random_state=42)
X_train_bal, y_train_bal = smote.fit_resample(X_train_scaled, y_train)

print("[2017] Applying PCA...")
pca = PCA(n_components=0.95, random_state=42)
X_train_pca = pca.fit_transform(X_train_bal)
X_test_pca = pca.transform(X_test_scaled)

print("[2017] Training Logistic Regression...")
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_pca, y_train_bal)

y_pred = lr_model.predict(X_test_pca)
print("\n[2017 Test Results]")
print(classification_report(y_test, y_pred))

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
cm_lr = confusion_matrix(y_test, y_pred)
sns.heatmap(cm_lr, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix - 2017 Test Set')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# --- ADDED: PRECISION-RECALL CURVE ---
print("[2017] Plotting Precision-Recall Curve...")
# Get probabilities for the positive class (Class 1: DDoS)
y_probs = lr_model.predict_proba(X_test_pca)[:, 1]
precision, recall, thresholds = precision_recall_curve(y_test, y_probs)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='purple', label='Logistic Regression')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve (CICIDS2017)')
plt.legend()
plt.grid(True)
plt.show()
# -------------------------------------


# ==========================================
# PHASE 2: CROSS-VALIDATION ON CICIDS2018
# ==========================================
print("\n--- PHASE 2: Cross-Validation on CICIDS2018 ---")

file_path_2018 = os.path.join(os.path.dirname(__file__), 'cicids2018.csv')

if os.path.exists(file_path_2018):
    df_2018 = pd.read_csv(file_path_2018)
    
    df_2018.columns = df_2018.columns.str.strip()
    df_2018.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_2018.dropna(inplace=True)
    df_2018.drop_duplicates(inplace=True)

    rename_dict = {
        'Dst Port': 'Destination Port',
        'Tot Fwd Pkts': 'Total Fwd Packets',
        'Tot Bwd Pkts': 'Total Backward Packets',
        'TotLen Fwd Pkts': 'Total Length of Fwd Packets',
        'TotLen Bwd Pkts': 'Total Length of Bwd Packets',
        'Fwd Pkt Len Max': 'Fwd Packet Length Max',
        'Fwd Pkt Len Min': 'Fwd Packet Length Min',
        'Fwd Pkt Len Mean': 'Fwd Packet Length Mean',
        'Fwd Pkt Len Std': 'Fwd Packet Length Std',
        'Bwd Pkt Len Max': 'Bwd Packet Length Max',
        'Bwd Pkt Len Min': 'Bwd Packet Length Min',
        'Bwd Pkt Len Mean': 'Bwd Packet Length Mean',
        'Bwd Pkt Len Std': 'Bwd Packet Length Std',
        'Flow Byts/s': 'Flow Bytes/s',
        'Flow Pkts/s': 'Flow Packets/s',
        'Fwd IAT Tot': 'Fwd IAT Total',
        'Bwd IAT Tot': 'Bwd IAT Total',
        'Fwd Header Len': 'Fwd Header Length',
        'Bwd Header Len': 'Bwd Header Length',
        'Fwd Pkts/s': 'Fwd Packets/s',
        'Bwd Pkts/s': 'Bwd Packets/s',
        'Pkt Len Min': 'Min Packet Length',
        'Pkt Len Max': 'Max Packet Length',
        'Pkt Len Mean': 'Packet Length Mean',
        'Pkt Len Std': 'Packet Length Std',
        'Pkt Len Var': 'Packet Length Variance',
        'FIN Flag Cnt': 'FIN Flag Count',
        'SYN Flag Cnt': 'SYN Flag Count',
        'RST Flag Cnt': 'RST Flag Count',
        'PSH Flag Cnt': 'PSH Flag Count',
        'ACK Flag Cnt': 'ACK Flag Count',
        'URG Flag Cnt': 'URG Flag Count',
        'ECE Flag Cnt': 'ECE Flag Count',
        'Pkt Size Avg': 'Average Packet Size',
        'Fwd Seg Size Avg': 'Avg Fwd Segment Size',
        'Bwd Seg Size Avg': 'Avg Bwd Segment Size',
        'Fwd Byts/b Avg': 'Fwd Avg Bytes/Bulk',
        'Fwd Pkts/b Avg': 'Fwd Avg Packets/Bulk',
        'Fwd Blk Rate Avg': 'Fwd Avg Bulk Rate',
        'Bwd Byts/b Avg': 'Bwd Avg Bytes/Bulk',
        'Bwd Pkts/b Avg': 'Bwd Avg Packets/Bulk',
        'Bwd Blk Rate Avg': 'Bwd Avg Bulk Rate',
        'Subflow Fwd Pkts': 'Subflow Fwd Packets',
        'Subflow Fwd Byts': 'Subflow Fwd Bytes',
        'Subflow Bwd Pkts': 'Subflow Bwd Packets',
        'Subflow Bwd Byts': 'Subflow Bwd Bytes',
        'Init Fwd Win Byts': 'Init_Win_bytes_forward',
        'Init Bwd Win Byts': 'Init_Win_bytes_backward',
        'Fwd Act Data Pkts': 'act_data_pkt_fwd',
        'Fwd Seg Size Min': 'min_seg_size_forward'
    }
    df_2018.rename(columns=rename_dict, inplace=True)

    if 'Fwd Header Length.1' in X.columns and 'Fwd Header Length.1' not in df_2018.columns:
        if 'Fwd Header Length' in df_2018.columns:
            df_2018['Fwd Header Length.1'] = df_2018['Fwd Header Length']
        else:
            df_2018['Fwd Header Length.1'] = 0


    df_2018['Label'] = df_2018['Label'].astype(str).apply(lambda x: 0 if x.lower() == 'benign' else 1)
    
    X_2018 = df_2018.drop('Label', axis=1)
    y_2018 = df_2018['Label']

    for col in X.columns:
        if col not in X_2018.columns:
            print(f"Warning: '{col}' missing in 2018. Filling with 0.")
            X_2018[col] = 0

    X_2018 = X_2018[X.columns]

    print("\n[VERIFICATION] Checking Feature Alignment...")
    if list(X.columns) == list(X_2018.columns):
        print("SUCCESS: Columns match exactly.")
    else:
        raise ValueError("FAILURE: Column mismatch despite fixes. Stopping.")

    print(f"[VERIFICATION] 2018 Labels: {np.unique(y_2018, return_counts=True)}")

    print("\n[2018] Transforming data...")
    X_2018_scaled = scaler.transform(X_2018)
    X_2018_pca = pca.transform(X_2018_scaled)
    
    print("[2018] Predicting...")
    y_pred_2018 = lr_model.predict(X_2018_pca)

    print(f"\n[2018 Cross-Dataset Results]")
    print(f"Accuracy on 2018: {accuracy_score(y_2018, y_pred_2018):.4f}")
    print(classification_report(y_2018, y_pred_2018))
    
    plt.figure(figsize=(8, 6))
    cm_2018 = confusion_matrix(y_2018, y_pred_2018)
    sns.heatmap(cm_2018, annot=True, fmt='d', cmap='Reds', cbar=False)
    plt.title('Confusion Matrix - 2018 Cross-Validation')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

else:
    print(f"File not found: {file_path_2018}")
    print("Please place 'cicids2018.csv' in the same folder to run Phase 2.")