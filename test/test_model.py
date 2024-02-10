import pickle
import pandas as pd
import numpy as np
import joblib
import json

boeke_model = joblib.load('model_boeke.sav')



data2 = [
    [3.93, 203.23, 31.27, 19.82, 21.47, 0, 1.79, 73582, 13.935, 121.42, 0, 1.635, 0.75375, 120.4392, 7.822670237],
    [5.87, 92.22, 26.7, 30.95, 11.52, 19583, 6.58, 71316, 9.828, 124.48, 0.274594761, 4.813333, 1.730833333, 153.3687, 21.94394411],
    [6.89, 337.3, 24.89, 36.09, 3.58, 0, 6.12, 70259, 9.765, 122.62, 0, 0.442857, 0.311309524, 354.4794, 26.77256496],
    [9.74, 620.04, 36, 47.67, 18.65, 0, 4.78, 59718, 10.377, 123.2, 0, 1.575, 0.617083333, 419.1316, 4.484601172],
    [11.22, 256.09, 44.1, 56.46, 22.2, 18312, 6.3, 136587, 9.3, 125.87, 0.134068396, 0.716667, 0.311458333, 264.727, 15.35290223],
    [0.63, 4.13, 2.27, 4.31, 2.63, 8964, 3.3, 26304, 17.535, 120.39, 0.340784672, 0.097222, 0.058156028, 139.8891, 6.750136725],
    [6.74, 80.29, 22.07, 35.34, 7.14, 91196, 8.24, 130546, 14.268, 122.49, 0.698573683, 1.18, 0.968085106, 14.94063, 12.18174604],
    [6.92, 232.73, 32.58, 35.34, 32.02, 0, 2.83, 100227, 15.334, 120.44, 0, 0.431944, 0.139007092, 109.2869, 4.089208628],
    [11.17, 199.57, 33.8, 56.7, 2.97, 6, 7.53, 84794, 11.288, 124.58, 0.00007076, 1.245, 0.463333333, 267.3691, 8.125925232]
]


# mean_slope=6.51
# meanelevation=182.04
# ruggedness=35.3
# meanruggedness=34.19
# area=9.61
# coastlength=6.32
# poverty=5240
# perimeter=100145
# glat=13.99
# glon=122.97
# coastperiratio=0.000063108
# rain6h=6.310417
# rain24h=3.888541667
# distrack=389.8607
# vmax=10.27861602

# data2=[mean_slope,meanelevation,ruggedness,meanruggedness,area,coastlength,poverty,
#perimeter,glat,glon,coastperiratio,rain6h,rain6h,distrack,vmax]


# input_data2 = np.array(data2)
# # Reshape the 1D array to a 2D array

# input_data_2d2 = input_data2.reshape(1, -1)

# # Make predictions using the loaded model

# prediction2 = boeke_model.predict(input_data_2d2)
with open('test/test.json', 'r') as json_file:
    data2 = json.load(json_file)

# print(prediction2)
for input_data in data2:
    input_data_np = np.array(input_data)
    # Reshape the 1D array to a 2D array
    input_data_2d = input_data_np.reshape(1, -1)

    # Make predictions using the loaded model
    prediction = boeke_model.predict(input_data_2d)

    print(round(prediction.item(),8))