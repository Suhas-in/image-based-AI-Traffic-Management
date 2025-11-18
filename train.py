import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import accuracy_score
import joblib
import os

# Load dataset
data = pd.read_csv('../data/traffic_data.csv')

# Feature engineering - create density ratio
data['density_ratio'] = data['vehicle_count'] / data['lane_capacity']

# Select features and label
X = data[['vehicle_count', 'avg_speed', 'density_ratio']]
y = data['congestion_level']

# Encode labels (Low, Medium, High → 0,1,2)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split dataset (70% train / 30% test)
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.3, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate accuracy
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {acc * 100:.2f}%")

# Create models folder if not exists
os.makedirs('../models', exist_ok=True)

# Save model, scaler, and label encoder
joblib.dump(model, '../models/rf_model.joblib')
joblib.dump(scaler, '../models/scaler.joblib')
joblib.dump(le, '../models/label_encoder.joblib')

print("✅ Model training complete! Files saved in /models/")
