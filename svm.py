from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
df = pd.read_csv("filtered data 3.csv")

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

# One-hot encode 'CAUSE'
if 'CAUSE' in df.columns:
    df = pd.get_dummies(df, columns=['CAUSE'], prefix='CAUSE')

# Normalizing 'Damage'
df['Damage'] = df['Damage'].replace({'Inaccessible': 'No Damage', 'inaccessible': 'No Damage'})

# One-hot encode for other categorical columns
one_hot_cols = {
    'Damage Inspection Date': 'DID',
    'Structure Type': 'ST'
}
for col, prefix in one_hot_cols.items():
    if col in df.columns:
        df = pd.get_dummies(df, columns=[col], prefix=prefix)

# Handling missing values in precipitation
if 'Max Observed Precipitation' in df.columns:
    df['Max Observed Precipitation'] = df['Max Observed Precipitation'].fillna(
        df['Max Observed Precipitation'].mean()
    )

# Label encode 'Damage'
le = LabelEncoder()
df['damage_label'] = le.fit_transform(df['Damage'])

# Dropping unnecessary columns for training
X = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0', 'month', 'Damage', 'damage_label'], errors='ignore')
y = df['damage_label']

# Splitting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training SVM with RBF kernel
svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)

# Evaluating model
accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred, average='weighted')
precision = precision_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

accuracy, recall, precision, f1
