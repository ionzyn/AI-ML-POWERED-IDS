import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
import joblib
import os

os.makedirs('models', exist_ok=True)

# Load both sets (training and testing)
train_df = pd.read_csv('data/UNSW_NB15_training-set.csv', na_values=['-', ' ', ''])
test_df  = pd.read_csv('data/UNSW_NB15_testing-set.csv',  na_values=['-', ' ', ''])

print("==================== INITIAL STATE ====================")
print(f"Train shape: {train_df.shape}")
print(f"Test shape:  {test_df.shape}")

# ============================================================
# Step 1: Separate metadata before cleaning
# ============================================================
print("\n==================== STEP 1: Separate Metadata ====================")

train_meta = train_df[['id', 'attack_cat']].copy()
test_meta  = test_df[['id', 'attack_cat']].copy()

train_df = train_df.drop(columns=['id', 'attack_cat', 'label'])  
test_df  = test_df.drop(columns=['id', 'attack_cat', 'label'])   

print(f"[✓] Metadata separated — train_meta shape: {train_meta.shape}, test_meta shape: {test_meta.shape}")
print(f"[✓] Train shape after drop: {train_df.shape}")
print(f"[✓] Test shape after drop:  {test_df.shape}")
print(f"[✓] Metadata columns: {train_meta.columns.tolist()}")
print(f"[✓] Remaining columns: {train_df.columns.tolist()}")

# ============================================================
# Step 2: Fill missing service values
# ============================================================
print("\n==================== STEP 2: Fix Missing Service Values ====================")

print(f"[Before] Train [service] nulls: {train_df['service'].isnull().sum()}")
print(f"[Before] Test [service] nulls:  {test_df['service'].isnull().sum()}")

train_df['service'] = train_df['service'].fillna('none')
test_df['service']  = test_df['service'].fillna('none')

known_services   = train_df['service'].unique().tolist()
test_df['service'] = test_df['service'].apply(lambda x: x if x in known_services else 'none')

print(f"[After]  Train [service] nulls: {train_df['service'].isnull().sum()}")
print(f"[After]  Test [service] nulls:  {test_df['service'].isnull().sum()}")
print("\n[✓] Train [service] value counts:")
print(train_df['service'].value_counts())

# ============================================================
# Step 3A: Group rare state values into 'other'
# ============================================================
print("\n==================== STEP 3A: Fix Rare State Values ====================")

state_counts = train_df['state'].value_counts()
rare_states  = state_counts[state_counts < 15].index.tolist()

print(f"[Before] Train [state] unique values: {train_df['state'].nunique()}")
print(f"[Info]   Rare [state]s being grouped: {rare_states}")

train_df['state'] = train_df['state'].replace(rare_states, 'other')

known_states    = train_df['state'].unique().tolist()
test_df['state'] = test_df['state'].apply(lambda x: x if x in known_states else 'other')

print(f"[After]  Train [state] unique values: {train_df['state'].nunique()}")
print("\n[✓] Train [state] value counts:")
print(train_df['state'].value_counts())

# ============================================================
# Step 3B: Group rare proto values into 'other'
# ============================================================
print("\n==================== STEP 3B: Fix Rare Proto Values ====================")

proto_counts = train_df['proto'].value_counts()
rare_protos  = proto_counts[proto_counts < 1000].index.tolist()

print(f"[Before] Train [proto] unique values: {train_df['proto'].nunique()}")
print(f"[Info]   Number of rare [proto]s being grouped: {len(rare_protos)}")
print(f"[Info]   Rare [proto]s: {rare_protos}")

train_df['proto'] = train_df['proto'].replace(rare_protos, 'other')

known_protos    = train_df['proto'].unique().tolist()
test_df['proto'] = test_df['proto'].apply(lambda x: x if x in known_protos else 'other')

print(f"[After]  Train [proto] unique values: {train_df['proto'].nunique()}")
print("\n[✓] Train [proto] value counts:")
print(train_df['proto'].value_counts())

# ============================================================
# Step 4: Cap Outliers (Winsorization at 99th percentile)
# ============================================================
print("\n==================== STEP 4: Cap Outliers ====================")

cols_to_cap = [
    'sbytes', 'dbytes',
    'sload',  'dload',
    'sjit',   'djit',
    'sinpkt', 'dinpkt',
    'spkts',  'dpkts',
    'sloss',  'dloss',
    'response_body_len',
    'stcpb',  'dtcpb',
]

for col in cols_to_cap:
    ceiling = train_df[col].quantile(0.99)

    print(
        f"[{col}] ceiling (99th percentile): {ceiling:.2f} | "
        f"train max before: {train_df[col].max():.2f}",
        end=""
    )

    train_df[col] = train_df[col].clip(upper=ceiling)
    test_df[col]  = test_df[col].clip(upper=ceiling)

    print(f" | train max after: {train_df[col].max():.2f}")

print("\n[✓] Outlier capping complete")
print(f"[✓] Final train shape: {train_df.shape}")
print(f"[✓] Final test shape:  {test_df.shape}")

# ============================================================
# Step 4B: Zero out features that cannot be extracted at inference
# ============================================================
print("\n==================== STEP 4B: Zero Out Unextractable Features ====================")

# These 5 features require application-layer parsing (HTTP/FTP) that Scapy
# cannot perform on raw packets. During live detection they are always 0.
# Zeroing them in training makes the model learn to classify WITHOUT relying
# on them, which directly improves confidence on live traffic.
LIVE_ZERO_FEATURES = [
    'trans_depth', 'response_body_len',
    'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd'
]

train_df[LIVE_ZERO_FEATURES] = 0
test_df[LIVE_ZERO_FEATURES]  = 0

print(f"[✓] Zeroed out {len(LIVE_ZERO_FEATURES)} unextractable features: {LIVE_ZERO_FEATURES}")

# ============================================================
# Step 5A: One-Hot Encode
# ============================================================
print("\n==================== STEP 5A: One-Hot Encoding ====================")

cat_cols = ['proto', 'service', 'state']

train_ohe = train_df.copy()
test_ohe = test_df.copy()

train_ohe = pd.get_dummies(train_ohe, columns=cat_cols)
test_ohe  = pd.get_dummies(test_ohe,  columns=cat_cols)

test_ohe = test_ohe.reindex(columns=train_ohe.columns, fill_value=0)

print(f"[✓] OHE train shape: {train_ohe.shape}")
print(f"[✓] OHE test shape:  {test_ohe.shape}")
print(f"[✓] Columns match:   {list(train_ohe.columns) == list(test_ohe.columns)}")

# ============================================================
# Step 5B: Label Encoding
# ============================================================
print("\n==================== STEP 5B: Label Encoding ====================")

train_le = train_df.copy()
test_le  = test_df.copy()

for col in cat_cols:
    le = LabelEncoder()

    le.fit(train_le[col])
    train_le[col] = le.transform(train_le[col])

    test_le[col] = test_le[col].map(
        lambda x: le.transform([x])[0] if x in le.classes_ else -1
    )
    joblib.dump(le, f'models/le_{col}.pkl')

# ============================================================
# Step 6: Separate Features and Target + Scale
# ============================================================
print("\n==================== STEP 6: Scaling ====================")

y_train = train_meta['attack_cat'].str.strip()
y_test  = test_meta['attack_cat'].str.strip()

print(f"[✓] Target class distribution (train):")
print(y_train.value_counts())

scaler_ohe = MinMaxScaler()
scaler_le  = MinMaxScaler()

X_train_ohe = scaler_ohe.fit_transform(train_ohe)
X_test_ohe  = scaler_ohe.transform(test_ohe)

X_train_le  = scaler_le.fit_transform(train_le)
X_test_le   = scaler_le.transform(test_le)

X_train_ohe = pd.DataFrame(X_train_ohe, columns=train_ohe.columns)
X_test_ohe  = pd.DataFrame(X_test_ohe,  columns=test_ohe.columns)
X_train_le  = pd.DataFrame(X_train_le,  columns=train_le.columns)
X_test_le   = pd.DataFrame(X_test_le,   columns=test_le.columns)

joblib.dump(scaler_ohe, 'models/scaler_ohe.pkl')
joblib.dump(scaler_le,  'models/scaler_le.pkl')

print(f"\n[✓] X_train_ohe shape: {X_train_ohe.shape}")
print(f"[✓] X_test_ohe shape:  {X_test_ohe.shape}")
print(f"[✓] X_train_le shape:  {X_train_le.shape}")
print(f"[✓] X_test_le shape:   {X_test_le.shape}")

print(f"\n[✓] OHE scaled min: {X_train_ohe.values.min():.4f}  max: {X_train_ohe.values.max():.4f}")
print(f"[✓] LE  scaled min: {X_train_le.values.min():.4f}  max: {X_train_le.values.max():.4f}")

# ============================================================
# Step 7: SMOTE (Class Imbalance)
# ============================================================
print("\n==================== STEP 7: SMOTE (Class Imbalance) ====================")

print(f"[Before SMOTE] Class distribution:")
print(y_train.value_counts())
print(f"\nWorms has only {y_train.value_counts()['Worms']} examples — the model will almost never detect them")

sampling_strategy = {
    'Analysis':  12000,
    'Backdoor':  12000,
    'Worms':     12000,
    'Shellcode': 12000,
}

smote = SMOTE(
    sampling_strategy=sampling_strategy,
    random_state=42,
    k_neighbors=3
)
X_train_smote, y_train_smote = smote.fit_resample(X_train_ohe, y_train)

X_train_smote = pd.DataFrame(X_train_smote, columns=X_train_ohe.columns)
y_train_smote = pd.Series(y_train_smote, name='attack_cat')

print(f"\n[After SMOTE] Class distribution:")
print(y_train_smote.value_counts())
print(f"\n[✓] X_train_smote shape: {X_train_smote.shape}")
print(f"[✓] All classes now have equal representation: {y_train_smote.value_counts().nunique() == 1}")

# ============================================================
# Step 8: Reattach Metadata
# ============================================================
print("\n==================== STEP 8: Reattach Metadata ====================")

# Reset indexes so row 0 of features aligns with row 0 of targets.
# Indexes can drift during transformations — reset gives clean sequential numbers.
y_train     = y_train.reset_index(drop=True)
y_test      = y_test.reset_index(drop=True)
X_train_ohe = X_train_ohe.reset_index(drop=True)
X_test_ohe  = X_test_ohe.reset_index(drop=True)
X_train_le  = X_train_le.reset_index(drop=True)
X_test_le   = X_test_le.reset_index(drop=True)

print(f"[✓] Indexes reset — y_train sample: {y_train.head(5).tolist()}")

# ============================================================
# Step 9: Save Cleaned Data to CSV
# ============================================================
print("\n==================== STEP 9: Save Cleaned Data to CSV ====================")

os.makedirs('data', exist_ok=True)

X_train_ohe.to_csv('data/X_train_ohe.csv', index=False)
X_test_ohe.to_csv('data/X_test_ohe.csv',   index=False)

X_train_le.to_csv('data/X_train_le.csv',   index=False)
X_test_le.to_csv('data/X_test_le.csv',     index=False)

X_train_smote.to_csv('data/X_train_smote.csv', index=False)

y_train.to_csv('data/y_train.csv',               index=False, header=True)
y_test.to_csv('data/y_test.csv',                 index=False, header=True)
pd.Series(y_train_smote).to_csv('data/y_train_smote.csv', index=False, header=True)

print("[✓] X_train_ohe    saved → data/X_train_ohe.csv")
print("[✓] X_test_ohe     saved → data/X_test_ohe.csv")
print("[✓] X_train_le     saved → data/X_train_le.csv")
print("[✓] X_test_le      saved → data/X_test_le.csv")
print("[✓] X_train_smote  saved → data/X_train_smote.csv")
print("[✓] y_train        saved → data/y_train.csv")
print("[✓] y_test         saved → data/y_test.csv")
print("[✓] y_train_smote  saved → data/y_train_smote.csv")
print("\n[✓] Data preparation complete — ready for model training")