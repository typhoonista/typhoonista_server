from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import requests

app = Flask(__name__)
CORS(app)

# Load the trained model
typhoonista_model = joblib.load('STACKED-SVR_RF_FINAL.joblib')
damagedArea_model = joblib.load('STACKED-SVR_RF-DAREA.joblib')
distance_model = joblib.load('STACKED-SVR_RF-DISTANCE.joblib')
damagedYield_model = joblib.load('STACKED-SVR_RF-DYIELD.joblib')
rainfall24_model = joblib.load('STACKED-SVR_RF-RAINFALL24.joblib')
rainfall6_model = joblib.load('STACKED-SVR_RF-RAINFALL6.joblib')
windspeed_model = joblib.load('STACKED-SVR_RF-WS.joblib')

def convert_to_serializable(data):
    # Convert non-serializable parts of the data to a serializable format
    if 'results' in data and data['results']:
        location = data['results'][0]['geometry']['location']
        latitude = location.get('lat', None)
        longitude = location.get('lng', None)

        return {"latitude": latitude, "longitude": longitude}
    else:
        return {"error": "No results found in the response."}

@app.route('/typhoonista/predict', methods=['POST'])
def predict1():
    try:
        data = request.get_json(force=True)
        input_data = np.array(data['features'])
        input_data_2d = input_data.reshape(1, -1)
        prediction = typhoonista_model.predict(input_data_2d)
        return str(prediction[0])  # Return the prediction value as a string
    except Exception as e:
        return str(e)  # Handle exceptions if necessary

@app.route('/damaged-area/predict', methods=['POST'])
def predict2():
    try:
        data = request.get_json(force=True)
        input_data = np.array(data['features'])
        input_data_2d = input_data.reshape(1, -1)
        prediction = damagedArea_model.predict(input_data_2d)
        return str(prediction[0])  # Return the prediction value as a string
    except Exception as e:
        return str(e)  # Handle exceptions if necessary
    
@app.route('/distance/predict', methods=['POST'])
def predict3():
    try:
        data = request.get_json(force=True)
        input_data = np.array(data['features'])
        input_data_2d = input_data.reshape(1, -1)
        prediction = distance_model.predict(input_data_2d)
        return str(prediction[0])  # Return the prediction value as a string
    except Exception as e:
        return str(e)  # Handle exceptions if necessary

@app.route('/damaged-yield/predict', methods=['POST'])
def predict4():
    try:
        data = request.get_json(force=True)
        input_data = np.array(data['features'])
        input_data_2d = input_data.reshape(1, -1)
        prediction = damagedYield_model.predict(input_data_2d)
        return str(prediction[0])  # Return the prediction value as a string
    except Exception as e:
        return str(e)  # Handle exceptions if necessary

@app.route('/rainfall24/predict', methods=['POST'])
def predict5():
    try:
        data = request.get_json(force=True)
        input_data = np.array(data['features'])
        input_data_2d = input_data.reshape(1, -1)
        prediction = rainfall24_model.predict(input_data_2d)
        return str(prediction[0])  # Return the prediction value as a string
    except Exception as e:
        return str(e)  # Handle exceptions if necessary

@app.route('/rainfall6h/predict', methods=['POST'])
def predict6():
    try:
        data = request.get_json(force=True)
        input_data = np.array(data['features'])
        input_data_2d = input_data.reshape(1, -1)
        prediction = rainfall6_model.predict(input_data_2d)
        return str(prediction[0])  # Return the prediction value as a string
    except Exception as e:
        return str(e)  # Handle exceptions if necessary

@app.route('/windspeed/predict', methods=['POST'])
def predict7():
    try:
        data = request.get_json(force=True)
        input_data = np.array(data['features'])
        input_data_2d = input_data.reshape(1, -1)
        prediction = windspeed_model.predict(input_data_2d)
        return str(prediction[0])  # Return the prediction value as a string
    except Exception as e:
        return str(e)  # Handle exceptions if necessary


@app.route('/get_coordinates', methods=['POST'])
def get_coordinates():
    api_url_geocode = "https://api-v2.distancematrix.ai/maps/api/geocode/json?"
    api_url_distance = "https://api-v2.distancematrix.ai/maps/api/distancematrix/json"
    api_key_geocode = "12gi8PRlBX15xzT44xBqmaTObXMklEZrp6imGFmvNJtgZFVVfloDvfZBnXuC9aZN"
    api_key_distance = "pAWX21Fc8J9hHS5petTMAYqZXQwmnoa5VBXGqf16S9aX2YMWER5m5kdgErqtP7Fk"

    origin_input = request.get_json().get('predictionLocation', '')
    destination_input = request.get_json().get('typhoonLocation', '')

    params_origin = {
        "address": origin_input,
        "key": api_key_geocode
    }
    origin_response = requests.get(api_url_geocode, params=params_origin)
    
    params_destination = {
        "address": destination_input,
        "key": api_key_geocode
    }
    destination_response = requests.get(api_url_geocode, params=params_destination)
    

    if origin_response.status_code == 200 and destination_response.status_code == 200:
        origin_lat = origin_response.json()['result'][0].get('geometry', {}).get('location', {}).get('lat', '')
        origin_lng = origin_response.json()['result'][0].get('geometry', {}).get('location', {}).get('lng', '')
        destination_lat = destination_response.json()['result'][0].get('geometry', {}).get('location', {}).get('lat', '')
        destination_lng = destination_response.json()['result'][0].get('geometry', {}).get('location', {}).get('lng', '')

        params_distance = {
            "origins": f"{origin_lat},{origin_lng}",
            "destinations": f"{destination_lat},{destination_lng}",
            "key": api_key_distance
}

        distance_response = requests.get(api_url_distance, params=params_distance)

        if distance_response.status_code == 200:
            distance_data = distance_response.json()
            print(distance_data)
            if 'destination_addresses' in distance_data and 'origin_addresses' in distance_data and 'rows' in distance_data:
                distance = distance_data['rows'][0]['elements'][0]['distance']['text']
                result = {"distance": distance}
                print(result)
                return jsonify(result)
            else:
                return jsonify({"error": "No results found in the distance API response."}), 400
        else:
            return jsonify({"error": f"Error in distance API: {distance_response.status_code}, {distance_response.text}"}), 500

    
if __name__ == '__main__':
    app.run(host= '0.0.0.0', port=5000)
    