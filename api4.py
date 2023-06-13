import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the dataset
dataset = pd.read_csv('final_test.csv')

# Preprocessing
df = dataset.dropna()
df['height'] = df['height'].astype(int)
df['age'] = df['age'].astype(int)
df = df[df['size'] != 'XXL']
df["size"].replace({"XXS": "XS", "XXXL": "XXL"}, inplace=True)

# Split data into features (x) and target (y)
x = df[['age', 'height', 'weight']]
y = df['size']

# Split data into training and testing sets
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# Model training
model = DecisionTreeClassifier()
model.fit(x_train, y_train)

# Define API endpoint for prediction
@app.route('/predict', methods=['GET'])
def predict():
    data = request.json
    age = data['age']
    height = data['height']
    weight = data['weight']
    input_features = np.array([[age, height, weight]])
    input_features_scaled = scaler.transform(input_features)
    prediction = model.predict(input_features_scaled)
    return jsonify({'prediction': prediction[0]})

# Define API endpoint for accuracy
@app.route('/accuracy', methods=['GET'])
def get_accuracy():
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return jsonify({'accuracy': accuracy})

# Define API endpoint for classification report
@app.route('/classification_report', methods=['GET'])
def get_classification_report():
    y_pred = model.predict(x_test)
    report = classification_report(y_test, y_pred)
    return jsonify({'classification_report': report})

if __name__ == '__main__':
    app.run(debug=True)


