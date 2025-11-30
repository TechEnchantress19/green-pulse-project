import os
import glob
import pandas as pd
import numpy as np
import kagglehub
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import classification_report, accuracy_score

# 1. Setup Directories
os.makedirs("models", exist_ok=True)

# 2. Download Dataset (UPDATED LINK)
print("⬇️  Downloading better dataset from Kaggle...")
# Using the standard 'atharvaingle' dataset which is more reliable
path = kagglehub.dataset_download("atharvaingle/crop-recommendation-dataset")

# --- SMART FILE SEARCH ---
print(f"🔍 Searching for CSV in: {path}")
# Find all CSV files recursively
all_csvs = glob.glob(os.path.join(path, "**", "*.csv"), recursive=True)

# Prioritize the main file 'Crop_recommendation.csv'
csv_path = next((f for f in all_csvs if "Crop_recommendation.csv" in f), None)

if not csv_path:
    if all_csvs:
        csv_path = all_csvs[0] # Fallback to first CSV if specific name not found
        print(f"⚠️ Exact file name not found. Using: {csv_path}")
    else:
        print(f"❌ ERROR: No CSV files found in {path}")
        exit()

# 3. Load Data
print(f"📂 Loading data from: {csv_path}")
df = pd.read_csv(csv_path)

# --- VERIFY COLUMNS ---
required_columns = ['N', 'P', 'K', 'temperature', 'humidity', 'ph', 'rainfall', 'label']
missing = [col for col in required_columns if col not in df.columns]

if missing:
    print(f"❌ ERROR: Dataset is missing columns: {missing}")
    print(f"Found columns: {list(df.columns)}")
    exit()

# 4. Data Preprocessing
print("✅ Data Structure Validated.")
X = df.drop('label', axis=1)
y = df['label']

# Encode Targets
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale Features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split Data
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 5. Define the Stacked Ensemble
print("🧠 Training Stacked Ensemble (RF + XGB + LGBM)...")
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ('xgb', XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss', random_state=42)),
    ('lgbm', LGBMClassifier(n_estimators=100, random_state=42, verbose=-1))
]

final_estimator = LogisticRegression()

clf = StackingClassifier(
    estimators=estimators,
    final_estimator=final_estimator,
    cv=StratifiedKFold(n_splits=5),
    n_jobs=-1
)

# 6. Train
clf.fit(X_train, y_train)

# 7. Evaluate
print("\n📊 --- MODEL PERFORMANCE REPORT ---")
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Overall Accuracy: {accuracy*100:.2f}%")

# 8. Save Everything
print("\n💾 Saving artifacts to 'models/'...")
joblib.dump(clf, "models/crop_model.joblib")
joblib.dump(le, "models/label_encoder.joblib")
joblib.dump(scaler, "models/scaler.joblib")
print("✅ Done! You can now run main.py")
