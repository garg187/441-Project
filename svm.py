from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("Filtered Data 3.csv")

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
    df = pd.get_dummies(df, columns=['CAUSE'], prefix='CAUSE')

df['Damage'] = df['Damage'].replace({'Inaccessible': 'No Damage', 'inaccessible': 'No Damage'})

one_hot_cols = {
    'Damage Inspection Date': 'DID',
    'Structure Type': 'ST'
}
for col, prefix in one_hot_cols.items():
    if col in df.columns:
        df = pd.get_dummies(df, columns=[col], prefix=prefix)

if 'Max Observed Precipitation' in df.columns:
    df['Max Observed Precipitation'] = df['Max Observed Precipitation'].fillna(
        df['Max Observed Precipitation'].mean()
    )

try:
    df['Lat_Long_Interaction'] = df.get('Latitude', 0) * df.get('Longitude', 0)
except KeyError:
    pass

if set(['FIRE_DURATION', 'GIS_ACRES']).issubset(df.columns):
    df['Duration_Per_Acre'] = df['FIRE_DURATION'].div(df['GIS_ACRES'].add(1e-6))

wind_col = 'Max Wind Speed'
if wind_col in df.columns:
    wind_bins = [-1, 10, 20, 30, 100]
    df['Wind_Bin'] = pd.cut(df[wind_col], bins=wind_bins, labels=range(len(wind_bins) - 1))


severity_order = {
    'No Damage': 0,
    'Affected (1-9%)': 1,
    'Minor (10-25%)': 2,
    'Major (26-50%)': 3,
    'Destroyed (>50%)': 4
}
df['damage_label'] = df['Damage'].map(severity_order)

X = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0', 'month', 'Damage', 'damage_label'], errors='ignore')
y = df['damage_label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM pipeline with scaling and grid search
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('svm', SVC(kernel='rbf', random_state=42))
])

param_grid = {
    'svm__C': [0.1, 1, 10],
    'svm__gamma': ['scale', 0.01, 0.001]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

best_svm = grid_search.best_estimator_
y_pred_svm = best_svm.predict(X_test)

# Evaluate improved SVM
svm_scores = {
    'accuracy': accuracy_score(y_test, y_pred_svm),
    'recall': recall_score(y_test, y_pred_svm, average='weighted'),
    'precision': precision_score(y_test, y_pred_svm, average='weighted'),
    'f1': f1_score(y_test, y_pred_svm, average='weighted'),
    'best_params': grid_search.best_params_
}

svm_scores
