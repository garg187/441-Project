import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import pickle

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

if 'Latitude' in df.columns and 'Longitude' in df.columns:
    df['Lat_Long_Interaction'] = df['Latitude'] * df['Longitude']
if 'FIRE_DURATION' in df.columns and 'GIS_ACRES' in df.columns:
    df['Duration_Per_Acre'] = df['FIRE_DURATION'] / (df['GIS_ACRES'] + 1e-6)
if 'Max Wind Speed' in df.columns:
    df['Wind_Bin'] = pd.cut(df['Max Wind Speed'], bins=[-1, 10, 20, 30, 100], labels=False)

print("Transformed DataFrame shape:", df.shape)

# encode and split
severity_order = {
    'No Damage': 0,
    'Affected (1-9%)': 1,
    'Minor (10-25%)': 2,
    'Major (26-50%)': 3,
    'Destroyed (>50%)': 4
}
df['damage_label'] = df['Damage'].map(severity_order)
X = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0', 'month', 'Damage', 'damage_label'])
y = df['damage_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# xgb classifier with randomized search cross-validation
sample_weights = compute_sample_weight(class_weight = 'balanced', y = y_train)

xgb_model = xgb.XGBClassifier(eval_metric = 'mlogloss', random_state = 42, n_jobs = -1)
params = {
    'n_estimators': [100, 200],
    'max_depth': [5, 10],
    'learning_rate': [0.1, 0.01],
    'subsample': [0.8, 1],
    'colsample_bytree': [0.8, 1],
    'gamma': [0, 0.1],
    'min_child_weight': [1, 3]
}

random_search = RandomizedSearchCV(
    xgb_model, param_distributions = params,
    n_iter = 40,
    scoring = 'accuracy',
    cv = 5, verbose = 1, random_state = 42, n_jobs = -1
)

random_search.fit(X_train, y_train, sample_weight = sample_weights)

best_xgb = random_search.best_estimator_
y_pred = best_xgb.predict(X_test)

# evaluate
print(f"Test Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"Recall (Sensitivity): {recall_score(y_test, y_pred, average = 'weighted'):.3f}")
print(f"Precision: {precision_score(y_test, y_pred, average = 'weighted'):.3f}")
print(f"F1 Score: {f1_score(y_test, y_pred, average = 'weighted'):.3f}")
print(f"Best Params: {random_search.best_params_}")

# save model
with open("xgb_model.pkl", "wb") as f:
    pickle.dump(best_xgb, f)

predictions = X_test.copy()
predictions['Actual'] = y_test
predictions['Predicted'] = y_pred

predictions.head(30)

# visualize
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from sklearn.model_selection import learning_curve
import numpy as np
from sklearn.model_selection import learning_curve

train_sizes, train_scores, val_scores = learning_curve(
    xgb_model, X, y, cv=5, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10), n_jobs=-1, shuffle=True, random_state=42
)

train_scores_mean = np.mean(train_scores, axis=1)
val_scores_mean = np.mean(val_scores, axis=1)
val_scores_std = np.std(val_scores, axis=1)

plt.plot(train_sizes, train_scores_mean, 'o-', color='blue', label="Training accuracy")
plt.plot(train_sizes, val_scores_mean, 'o-', color='green', label="Validation accuracy")
plt.fill_between(train_sizes, val_scores_mean - val_scores_std, val_scores_mean + val_scores_std,
                 alpha=0.2, color='green')

plt.title("Learning Curve - XGBoost")
plt.xlabel("Training Set Size")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend(loc="lower right")
plt.tight_layout()
plt.show()

labels = [
    "No Damage",
    "Affected (1-9%)",
    "Minor (10-25%)",
    "Major (26-50%)",
    "Destroyed (>50%)"
]

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)

plt.figure(figsize=(10, 8))
disp.plot(cmap=plt.cm.Blues, xticks_rotation='vertical')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("XGBoost Confusion Matrix")
plt.tight_layout()
plt.show()
