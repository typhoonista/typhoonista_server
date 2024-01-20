from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np

app = Flask(__name__)
CORS(app)

# Load the trained model
model1 = joblib.load('SVR_RF_STACKED.joblib')
model2 = joblib.load('model_boeke.sav')

@app.route('/typhoonista/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        input_data = np.array(data['features'])
        input_data_2d = input_data.reshape(1, -1)
        prediction = model1.predict(input_data_2d)
        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/boeke/predict', methods=['POST'])
def predict2():
    try:
        print(model2)
        data = request.get_json(force=True)
        input_data = np.array(data['features'])
        input_data_2d = input_data.reshape(1, -1)
        prediction = model2.predict(input_data_2d)
        return jsonify({"prediction": prediction.tolist()})

    except Exception as e:
        return jsonify({"error": str(e)})

@app.route('/get_coordinates', methods=['GET'])
def get_coordinates():
    location_input = request.args.get('location_input', '')

    api_url = "https://api-v2.distancematrix.ai/maps/api/geocode/json?"
    api_key = "12gi8PRlBX15xzT44xBqmaTObXMklEZrp6imGFmvNJtgZFVVfloDvfZBnXuC9aZN"

    params = {
        "address": location_input,
        "key": api_key
    }

    response = request.get(api_url, params=params)

    if response.status_code == 200:
        data = response.json()

        if 'results' in data and data['results']:
            location = data['results'][0]['geometry']['location']
            latitude = location['lat']
            longitude = location['lng']

            result = {"latitude": latitude, "longitude": longitude}
            return jsonify(result)
        else:
            return jsonify({"error": "No results found in the response."}), 400
    else:
        return jsonify({"error": f"Error: {response.status_code}, {response.text}"}), 500

    
if __name__ == '__main__':
    app.run(host= '0.0.0.0', port=5000)
    