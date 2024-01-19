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
    
if __name__ == '__main__':
    app.run(host= '0.0.0.0', port=5000)
    