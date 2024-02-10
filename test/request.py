import requests
import json

data2 = {"features": [12,190,24,10,3285,1]}



response2 = requests.post("http://127.0.0.1:5000/windspeed/predict", json=data2)
# print("boeke tanginamo:",response1.json())
print(response2.json())

