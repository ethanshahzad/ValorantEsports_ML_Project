import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib



script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "dataset.csv")

# 1. Load dataset
df = pd.read_csv(csv_path)
print("Before filtering:", df.shape)

# 2. Drop rows with < 10 matches for either team
df = df[(df["teamA_matches_used"] >= 10) & (df["teamB_matches_used"] >= 10)]
print("After filtering:", df.shape)

# 3. Create features and labels
X = df.drop(columns=["winner", "link", "date", "teamA_matches_used", "teamB_matches_used"])
# y = df["winner"]

X= X[[c for c in X.columns if c.startswith("diff_")]]
y = df["winner"]

# Split 80% training / 20% testing
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,      
    random_state=1234,    
    stratify=y          
)

print("Training set size:", X_train.shape)
print("Test set size:", X_test.shape)

# ------------------------------------------------------------------------------------------------------------------------
# Testing Random Forest Model:

model = RandomForestClassifier(n_estimators=100, random_state=1234)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

importances = model.feature_importances_
features = X.columns

feat_importance = pd.DataFrame({
    "feature": features,
    "importance": importances
}).sort_values(by="importance", ascending=False)

print("\nFeature importances:\n", feat_importance.to_string(index=False))



# ------------------------------------------------------------------------------------------------------------------------
# Testing GridCV Model:

# param_grid = {
#     'n_estimators': [100, 200, 300],
#     'max_depth': [None, 10, 20],
#     'min_samples_split': [2, 5, 10]
# }

# grid = GridSearchCV(
#     RandomForestClassifier(random_state=1234),
#     param_grid,
#     cv=5,
#     scoring='accuracy'
# )
# grid.fit(X_train, y_train)

# print("Best params:", grid.best_params_)
# print("Best CV score:", grid.best_score_)