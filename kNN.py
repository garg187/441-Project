import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# read data
df = pd.read_csv('filtered_data3.csv')

# Drop columns not relevant to kNN
df = df.drop(columns=['Unnamed: 0.1', 'Unnamed: 0', '0', 'D0', 'D1', 'D2', 'D3', 'D4', 'W0', 'W1', 'W2', 'W3', 'W4',
                      'INC_NUM', 'ALARM_DATE', 'CONT_DATE', 'Damage Inspection Date', 'Fuel Moisture Date'])

# Note: no rows where damage missing

# fill in missing values for numeric columns (mean? median? mode?)
# fill with mean: Observed Max Temp, Observed Min Temp, Median Observed Temp, Max Wind Speed, Latitude, Longitude,
# GIS_ACRES, Shape__Area, Shape__Length, Max Observed Precipitation
mean_col = ['Observed Max Temp', 'Observed Min Temp', 'Median Observed Temp', 'Max Wind Speed', 'Latitude', 'Longitude',
            'GIS_ACRES', 'Shape__Area', 'Shape__Length', 'Max Observed Precipitation']
df[mean_col] = df[mean_col].fillna(df[mean_col].mean())
# fill with median: PERCENT, FIRE_DURATION
median_col = ['PERCENT', 'FIRE_DURATION']
df[median_col] = df[median_col].fillna(df[median_col].median())
# fill with mode: Zip Code,
mode_zip = float(df['Zip Code'].mode())
df['Zip Code'] = df['Zip Code'].fillna(mode_zip)

# encode categorical columns
cat_cols = df.select_dtypes(include=['object']).columns.drop('Damage')
df[cat_cols] = df[cat_cols].apply(LabelEncoder().fit_transform)

# Encode target column
df['Damage'] = LabelEncoder().fit_transform(df['Damage'])

# split data from target
X = df.drop(columns=['Damage'])
y = df['Damage']

# standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# determine best k-value for range(1, sqrt(n))
k_values = list(range(1, 53))
cv_scores = []
for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, X_scaled, y, cv = 5, scoring='accuracy')
    cv_scores.append(scores.mean())

best_k = k_values[cv_scores.index(max(cv_scores))]
print(best_k)

# train kNN classifier
kNN = KNeighborsClassifier(n_neighbors=best_k)
kNN.fit(X_train, y_train)
# 5-fold cross validation
score = cross_val_score(kNN, X_scaled, y, cv=3, scoring='accuracy')
print(score)

# Predictions
y_pred = kNN.predict(X_test)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
