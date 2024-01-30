import pandas as pd

# # Load the Excel file into a pandas DataFrame
# excel_file = 'municipalities.xlsx'
# df = pd.read_excel(excel_file, sheet_name='Sheet1')  # Replace 'Sheet1' with the actual sheet name

# # Convert DataFrame to JSON and save it to a file
# json_file = 'output.json'
# df.to_json(json_file, orient='records', lines=True)


json_file = 'output.json'

# Read the JSON file into a pandas DataFrame
df = pd.read_json(json_file, lines=True)

# Display the DataFrame
print(df)