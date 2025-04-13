import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pickle

pd.set_option('display.max_columns', None)
df = pd.read_csv("filtered_data3.csv")

# 1. Normalize 'year': 2000 → 0, 2001 → 1, ...
if 'year' in df.columns:
    df['year'] = df['year'] - 2000

# 2. Drop specified columns
drop_cols = [
    'Observed Max Temp', 'Observed Min Temp', 'INC_NUM',
    'ALARM_DATE', 'CONT_DATE', 'Zip Code', 'Fuel Moisture Date', 'Drought_Index_Cat'
]
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

# 3. Fill missing values in specific columns with column mean
for col in ['Median Observed Temp', 'PERCENT', 'Latitude', 'Longitude']:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mean())

# 4. Fill specified columns with 0 where missing
fill_zero_cols = [
    '0', 'D0', 'D1', 'D2', 'D3', 'D4',
    'W0', 'W1', 'W2', 'W3', 'W4',
    'Max Wind Speed', 'FIRE_DURATION',
    'GIS_ACRES', 'Shape__Area', 'Shape__Length'
]
for col in fill_zero_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# 5. One-hot encode 'CAUSE'
if 'CAUSE' in df.columns:
    df = pd.get_dummies(df, columns=['CAUSE'], prefix='CAUSE')

df['Damage'] = df['Damage'].replace({'Inaccessible': 'No Damage', 'inaccessible': 'No Damage'})

# 6. One-hot encode 'Damage Inspection Date', 'Damage', 'Structure Type'
# Use different prefixes to avoid overlap
one_hot_cols = {
    'Damage Inspection Date': 'DID',
    'Structure Type': 'ST'
}
for col, prefix in one_hot_cols.items():
    if col in df.columns:
        df = pd.get_dummies(df, columns=[col], prefix=prefix)

df['Max Observed Precipitation'] = df['Max Observed Precipitation'].fillna(df['Max Observed Precipitation'].mean())


# Final result
print("Transformed DataFrame shape:", df.shape)

le = LabelEncoder()
df['damage_label'] = le.fit_transform(df['Damage'])
X = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0', 'month', 'Damage', 'damage_label'])  # Replace with your actual feature selection
y = df['damage_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 15],
    'max_features': ['sqrt', 'log2', 0.5]
}

coarse_grid = GridSearchCV(
    RandomForestClassifier(random_state=42),
    param_grid=param_dist,
    cv=5,
    n_jobs=-1,
    verbose=1
)
coarse_grid.fit(X_train, y_train)

best_coarse = coarse_grid.best_params_
print("Best Coarse Parameters:", best_coarse)

best_fine_model = coarse_grid.best_estimator_
y_pred = best_fine_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)

print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Recall (Sensitivity): {recall_score(y_test, y_pred, average='weighted'):.3f}")
print(f"Precision: {precision_score(y_test, y_pred, average='weighted'):.3f}")
print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.3f}")

with open('random_forest_model.pkl', 'wb') as f:
    pickle.dump(best_fine_model, f)