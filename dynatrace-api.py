# Import necessary libraries
import requests  # For API requests
import json  # For handling JSON data
import pandas as pd  # For working with dataframes
import numpy as np  # For numerical operations
from datetime import datetime
import warnings  # To suppress warnings

# Disable unnecessary warnings
warnings.filterwarnings('ignore')

# Dynatrace API configuration
headers = {
    'Content-Type': 'application/json',
    'Accept': 'application/json',
    'Authorization': '***'
}

# Base Dynatrace API URL
base_url = "https://upc02393.live.dynatrace.com/api/v2/metrics/query"

# Define the time range for fetching data
start_time = datetime.strptime("2025-04-26T07:30:00.000Z", "%Y-%m-%dT%H:%M:%S.%fZ").strftime('%Y-%m-%dT%H:%M:%S.000Z')
end_time = datetime.strptime("2025-04-26T09:30:00.000Z", "%Y-%m-%dT%H:%M:%S.%fZ").strftime('%Y-%m-%dT%H:%M:%S.000Z')


# List of host IDs (Can be modified dynamically)
entity_ids = ["HOST-115AF0E56D335A15"]

# List of metric types to fetch (Dynamically added)
metric_types = [
    "builtin:host.cpu.usage",
    "builtin:host.mem.usage",
    "builtin:host.disk.usedPct",
    "builtin:host.disk.readOps",
    "builtin:host.disk.writeOps"
]

# Format metric selector and entity selector dynamically
metric_selector = ",".join(metric_types)
entity_selector = "entityId(" + ",".join(f'"{host}"' for host in entity_ids) + ")"

# Construct API request URL dynamically
query_url = f"{base_url}?metricSelector={metric_selector}&entitySelector={entity_selector}&from={start_time}&to={end_time}&resolution=1m"

# Make the API request
response = requests.get(query_url, headers=headers, verify=False)

# Initialize a set to track unique entries
seen = set()
data = []  # Initialize an empty list to store the data

if response.status_code == 200:
    result = response.json()  # Parse the JSON response

    # Extract relevant information from the API result
    for data_item in result.get('result', []):
        metricid = data_item['metricId']
        for item in data_item['data']:
            metricPath = metricid
            hostId = item['dimensions'][0]
            valuesList = item['values']
            timestampsList = item['timestamps']

            # Iterate over timestamps and values
            for i in range(len(timestampsList)):
                timestamps = timestampsList[i]
                values = valuesList[i]

                # Convert Unix timestamp to readable datetime format
                timestamp_dt = pd.to_datetime(timestamps, unit='ms')

                # Create a unique identifier for each data entry
                entry_key = (hostId, timestamp_dt, metricPath, values)

                # Store only unique entries
                if entry_key not in seen:
                    seen.add(entry_key)  # Mark as seen
                    data.append({
                        'host': hostId,
                        'timestamp': timestamp_dt,
                        'value': values,
                        'metricType': metricPath
                    })
else:
    print("API Error:", response.status_code, response.text)

# Convert collected data into a Pandas DataFrame
df = pd.DataFrame(data)

# Ensure timestamp is localized to UTC and converted to IST (Asia/Kolkata)
df['timestamp'] = df['timestamp'].dt.tz_localize('UTC').dt.tz_convert('Asia/Kolkata').dt.tz_localize(None)


# Filter data for specific host (optional)
df = df[df['host'] == "HOST-115AF0E56D335A15"]


metric_types = {
    "builtin:host.cpu.usage": 'CPU Usage %',
    "builtin:host.mem.usage": 'Memory Usage %',
    "builtin:host.disk.usedPct": 'Disk Usage %',
    "builtin:host.disk.readOps": 'readOps Usage %',
    "builtin:host.disk.writeOps":'writeOps Usage %'
}
df['metricType'] = df['metricType'].map(metric_types)

print("Data retrieval complete!")
df = df.groupby(['host', 'timestamp', 'metricType'], as_index=False).mean()
# Save processed data to CSV
df.to_csv('dynatrace_2604.csv', index=False)
print(df.head())



