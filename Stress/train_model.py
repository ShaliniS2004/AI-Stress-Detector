import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Create sample data if file doesn't exist
if not os.path.exists("stress_data.csv"):
    data = {
        'age': [20, 19, 21, 20, 22, 23, 19, 20, 21, 22],
        'study_hours': [5, 3, 6, 4, 2, 7, 4, 5, 3, 6],
        'sleep_hours': [6, 7, 5, 6, 5, 8, 7, 6, 5, 7],
        'physical_activity': [2, 4, 1, 3, 2, 5, 3, 2, 4, 3],
        'social_support': [3, 4, 2, 4, 1, 5, 3, 4, 2, 4],
        'stress_level': ['High', 'Low', 'High', 'Low', 'High', 'Low', 'Medium', 'High', 'Medium', 'Low']
    }
    df = pd.DataFrame(data)
    df.to_csv("stress_data.csv", index=False)
else:
    df = pd.read_csv("stress_data.csv")

# Encode the target variable
le = LabelEncoder()
df['stress_level'] = le.fit_transform(df['stress_level'])

# Split data
X = df.drop('stress_level', axis=1)
y = df['stress_level']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model and encoder
try:
    joblib.dump(model, "stress_model.pkl")
    joblib.dump(le, "label_encoder.pkl")
    print("Model and encoder saved successfully!")
except Exception as e:
    print(f"Error saving files: {e}")
    