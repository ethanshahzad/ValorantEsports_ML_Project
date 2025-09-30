import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
import xgboost as xgb
import numpy as np

# Load dataset
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "dataset_updated.csv")
df = pd.read_csv(csv_path)

df = df[(df["teamA_matches_used"] >= 10) & (df["teamB_matches_used"] >= 10)]

# ------------------------------------------------------------------------------------------------------------------------
# Feature engineering
df['weighted_recent_diff'] = (
    0.7 * (df['teamA_recent5_winrate'] - df['teamB_recent5_winrate']) +
    0.3 * (df['teamA_winrate'] - df['teamB_winrate'])
)

X = df[[c for c in df.columns if c.startswith("diff_")] + ['weighted_recent_diff'] + ['chat_teamA_conf']]
y = df["winner"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=1234, stratify=y
)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

# ------------------------------------------------------------------------------------------------------------------------
# Testing XGBoost Model:
model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    random_state=1234,
    use_label_encoder=False,
    eval_metric='logloss'
)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", round(accuracy, 4))

importances = model.feature_importances_
features = X.columns
feat_importance = pd.DataFrame({
    "feature": features,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nFeature importances:\n", feat_importance.to_string(index=False))