import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest

# X_train = pd.read_csv('train_features.csv')
# X_train_numeric = X_train.drop(columns=['anomaly'], errors='ignore').select_dtypes(include=['number'])
# X_train_numeric = X_train_numeric.fillna(X_train_numeric.mean())

# model = IsolationForest(contamination=0.05, random_state=42)
# model.fit(X_train_numeric)
# joblib.dump(model, 'isolation_forest_lead_model.pkl')

model = joblib.load('isolation_forest_lead_model.pkl')


X_test = pd.read_csv('test_features.csv')
X_test_numeric = X_test.drop(columns=['row_id'], errors='ignore').select_dtypes(include=['number'])
X_test_numeric = X_test_numeric.fillna(X_test_numeric.mean())
predictions = model.predict(X_test_numeric)

X_test['is_anomaly'] = predictions
print(X_test.head())

binary_preds = [1 if x == -1 else 0 for x in predictions]
total_rows = len(binary_preds)
anomaly_count = sum(binary_preds)
percentage = (anomaly_count / total_rows) * 100

print(f"Total rows: {total_rows}")
print(f"Anomalies found: {anomaly_count}")
print(f"Anomaly Percentage: {percentage:.2f}%")