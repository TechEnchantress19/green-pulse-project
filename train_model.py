import os
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

# 2. Download Dataset
print("⬇️  Downloading dataset from Kaggle...")
path = kagglehub.dataset_download("irakozekelly/crop-recommendation-dataset")
csv_path = os.path.join(path, "Crop_recommendation.csv")

# 3. Load Data
print(f"📂 Loading data from: {csv_path}")
df = pd.read_csv(csv_path)
X = df.drop('label', axis=1)
y = df['label']

# 4. Data Preprocessing (CRITICAL FOR PRECISION)
# Encode Targets (Rice -> 0, Maize -> 1)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Scale Features (CRITICAL for the Meta-Model)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split Data (Stratified to maintain class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# 5. Define the Stacked Ensemble
print("🧠 Training Stacked Ensemble (RF + XGB + LGBM)...")

# Level 0: Base Learners
estimators = [
    ('rf', RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)),
    ('xgb', XGBClassifier(n_estimators=100, use_label_encoder=False, eval_metric='mlogloss', random_state=42)),
    ('lgbm', LGBMClassifier(n_estimators=100, random_state=42, verbose=-1))
]

# Level 1: Meta-Learner
# We use Logistic Regression to find the best combination of the base models.
final_estimator = LogisticRegression()

clf = StackingClassifier(
    estimators=estimators,
    final_estimator=final_estimator,
    cv=StratifiedKFold(n_splits=5), # 5-Fold Cross Validation for robustness
    n_jobs=-1
)

# 6. Train
clf.fit(X_train, y_train)

# 7. Evaluate Precision
print("\n📊 --- MODEL PERFORMANCE REPORT ---")
y_pred = clf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Overall Accuracy: {accuracy*100:.2f}%")
print("\nDetailed Report:")
# Decode labels back to strings for the report (0 -> Rice)
target_names = [str(cls) for cls in le.classes_]
print(classification_report(y_test, y_pred, target_names=target_names))

# 8. Save Everything
print("\n💾 Saving artifacts...")
joblib.dump(clf, "models/crop_model.joblib")
joblib.dump(le, "models/label_encoder.joblib")
joblib.dump(scaler, "models/scaler.joblib") # Save scaler to use in main.py
print("Done! You can now run main.py")
