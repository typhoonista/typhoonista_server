import requests
location_input = "Abra de ilog, philippines"

api_url = "https://api-v2.distancematrix.ai/maps/api/geocode/json?"
address = location_input
api_key = "12gi8PRlBX15xzT44xBqmaTObXMklEZrp6imGFmvNJtgZFVVfloDvfZBnXuC9aZN"

params = {
    "address": address,
    "key": api_key
}

response = requests.get(api_url, params=params)

if response.status_code == 200:
    data = response.json()
    print(data)
else:
    print(f"Error: {response.status_code}, {response.text}")
