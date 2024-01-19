import requests

data = {"features": [23,23,45,45324,65,34,2345,23,23,2,1,2,3,4,5]}
data2 = {"features": [23,23,45,45324,65,34,2345]}


response1 = requests.post("http://localhost:5000/boeke/predict", json=data)
response2 = requests.post("http://localhost:5000/typhoonista/predict", json=data2)
print("boeke tanginamo:",response1.json())
print("typhoonista i love you",response2.json())
