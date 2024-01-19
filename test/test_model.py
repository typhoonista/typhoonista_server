import pickle
import pandas as pd
import numpy as np
import joblib

typhoonista_model = joblib.load('models\STACKED-SVR_RF102.joblib')
boeke_model = joblib.load('models\model_boeke.sav')

data1 = [1,3,2,3,4,2,4]
data2 = [1,3,4,2,5,2,4,6,3,2,2,3,4,1,2]
input_data1 = np.array(data1)
input_data2 = np.array(data2)
# Reshape the 1D array to a 2D array
input_data_2d1 = input_data1.reshape(1, -1)
input_data_2d2 = input_data2.reshape(1, -1)

# Make predictions using the loaded model
prediction1 = typhoonista_model.predict(input_data_2d1)
prediction2 = boeke_model.predict(input_data_2d2)

print(prediction1)
print(prediction2)