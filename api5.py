from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load the model from the file
model = joblib.load('model_decision_tree.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the JSON data from the request
    data = request.get_json()

    # Extract the features from the JSON data
    age = data['age']
    height = data['height']
    weight = data['weight']

    # Reshape the data to match the expected format
    reshaped_data = np.array([[age, height, weight]])

    # Make the prediction using the reshaped data
    prediction = model.predict(reshaped_data)

    # Send the prediction result as a response
    response = {'prediction': prediction[0]}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
