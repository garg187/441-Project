import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
import pickle
import xgboost as xgb

# data loading and preprocessing
df = pd.read_csv("filtered_data3.csv")

if 'year' in df.columns:
    df['year'] = df['year'] - 2000

drop_cols = [
    'Observed Max Temp', 'Observed Min Temp', 'INC_NUM',
    'ALARM_DATE', 'CONT_DATE', 'Zip Code', 'Fuel Moisture Date', 'Drought_Index_Cat'
]
df = df.drop(columns=[col for col in drop_cols if col in df.columns])

for col in ['Median Observed Temp', 'PERCENT', 'Latitude', 'Longitude']:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mean())

fill_zero_cols = [
    '0', 'D0', 'D1', 'D2', 'D3', 'D4',
    'W0', 'W1', 'W2', 'W3', 'W4',
    'Max Wind Speed', 'FIRE_DURATION',
    'GIS_ACRES', 'Shape__Area', 'Shape__Length'
]
for col in fill_zero_cols:
    if col in df.columns:
        df[col] = df[col].fillna(0)

if 'CAUSE' in df.columns:
    df = pd.get_dummies(df, columns = ['CAUSE'], prefix = 'CAUSE')

df['Damage'] = df['Damage'].replace({'Inaccessible': 'No Damage', 'inaccessible': 'No Damage'})

one_hot_cols = {
    'Damage Inspection Date': 'DID',
    'Structure Type': 'ST'
}
for col, prefix in one_hot_cols.items():
    if col in df.columns:
        df = pd.get_dummies(df, columns = [col], prefix = prefix)

df['Max Observed Precipitation'] = df['Max Observed Precipitation'].fillna(df['Max Observed Precipitation'].mean())

print("Transformed DataFrame shape:", df.shape)

# encode and split
le = LabelEncoder()
df['damage_label'] = le.fit_transform(df['Damage'])
X = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0', 'month', 'Damage', 'damage_label'])
y = df['damage_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# xgb classifier
xgb = xgb.XGBClassifier(eval_metric = 'mlogloss', random_state = 42)
params = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'learning_rate': [0.1, 0.01],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1]
}

xgb_grid = GridSearchCV(xgb, params, cv = 5, n_jobs = -1, verbose = 1)
xgb_grid.fit(X_train, y_train)

best_xgb = xgb_grid.best_estimator_
y_pred = best_xgb.predict(X_test)

# evaluate
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Recall (Sensitivity): {recall_score(y_test, y_pred, average = 'weighted'):.3f}")
print(f"Precision: {precision_score(y_test, y_pred, average = 'weighted'):.3f}")
print(f"F1 Score: {f1_score(y_test, y_pred, average = 'weighted'):.3f}")
print(f"Best Parameters: {xgb_grid.best_params_}")

# save model
with open("xgb_model.pkl", "wb") as f:
    pickle.dump(best_xgb, f)

# map damage level encoded values
dict(zip(le.transform(le.classes_), le.classes_))

# predictions
predictions = X_test.copy()
predictions['Actual'] = y_test
predictions['Predicted'] = y_pred

print(predictions.head(30))