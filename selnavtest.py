import joblib
import pandas as pd
from pmdarima import auto_arima
import requests
import pandas as pd
import json
from datetime import datetime, timedelta
import mysql.connector
import warnings 
import sqlalchemy
from urllib.parse import quote_plus, urlparse
 
import smtplib 
from email.mime.text import MIMEText
import importlib
import pandas as pd
import numpy as np
# from pmdarima import auto_arima
from statsmodels.tsa.seasonal import seasonal_decompose
import mysql.connector
import sqlalchemy
from urllib.parse import quote_plus, urlparse
import pandas as pd
import smtplib 
from email.mime.text import MIMEText
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix
from pmdarima import auto_arima
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pmdarima import auto_arima
# from keras.models import Sequential
# from keras.layers import LSTM, Dense, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score
from pmdarima import auto_arima

from sklearn.metrics import confusion_matrix 
warnings.filterwarnings('ignore')

# Resolution for the metrics
resolution = "1m"  # 1 minute resolution

seen = set()
data = []  # Ensure data is initialized as an empty list
 
for entity_id in entity_ids:
    for metric in metric_types:
        # Build the query parameters
        params = {
            # "metricSelector": f"builtin:{metric}:avg",
            # "entitySelector": f"entityId({entity_id})",
            "from": start_time,
            "to": end_time,
            # "resolution": resolution
        }
        # Make the API request
        response = requests.get(dynatrace_api_url, headers=headers, params=params)
        if response.status_code == 200:
            result = response.json()
            # print('result', result)
 
            for data_item in result.get('result', []):
                metricid = data_item['metricId']
                for item in data_item['data']:
                    metricPath = metricid
                    if 'cpu' in metricPath:
                        metricType = 'CPU Usage %'
                    elif 'mem' in metricPath:
                        metricType = 'Memory Usage %'
                    elif 'disk.usedPct' in metricPath:
                        metricType = 'Disk Usage %'
                    elif 'disk.readOps' in metricPath:
                        metricType = 'readOps Usage %'    
                    elif 'disk.writeOps' in metricPath:
                        metricType = 'writeOps Usage %'    
                    hostId = item['dimensions'][0]
                    valuesList = item['values']
                    timestampsList = item['timestamps']
 
                    for i in range(len(timestampsList)):
                        timestamps = timestampsList[i]
                        values = valuesList[i]
 
                        # Create a unique identifier for this data entry
                        entry_key = (hostId, timestamps, metricType, values)
 
                        # Check if this entry has been seen before
                        if entry_key not in seen:
                            seen.add(entry_key)  # Add to the set of seen entries
                            # Add the entry to the data list
                            data.append({
                                'host': hostId,
                                'timestamp': timestamps,
                                'value': values,
                                'metricType': metricType,
                                'CI_Tpye': 'Host'
                            })
        else:
            print('API Error')
 
# Convert the list of dictionaries into a pandas DataFrame
df = pd.DataFrame(data)
# print(df.tail())
# print('Before:',len(df)) 
# Remove duplicates based on relevant columns ('host', 'timestamp', 'metricType', 'value', 'CI_Tpye')
df = df.drop_duplicates(subset=['host', 'timestamp', 'metricType'],keep='first')
df=df.dropna() 
# print(df['metricType'].unique()) 
# print(df['host'].unique()) 
# Now, df is a DataFrame with no duplicates
# print(df.tail())
# print('After:',len(df))
last_1_day_data = pd.DataFrame({
    'timestamp': pd.date_range(start=end_time, periods=1, freq='min'),
    # 'value': [0] * 1440  # Replace this with actual values (e.g., your time series data)
})
# df.to_sql(name='selnavfinaldata', con=conn, if_exists='append', index=False)
# print('Completed') 
group_data=df
    # Check if 'value' column exists

group_data = group_data.sort_values(by='timestamp')  
group_data = group_data.reset_index(drop=True)
# print('dataframe column names:',df.columns)
def detect_anomalies(group_data):
    # Check if 'value' column exists
    if 'value' not in group_data.columns:
        print("Error: 'value' column not found in the data.")
        return
 
    # Fit ARIMA model
    # print(group_data.columns)
    # print(len(group_data))
    # print(group_data['value'].dtype)
 
    if group_data['value'].isnull().any():
        # print("Warning: NaN values found in 'value' column.")
        # Option 1: Drop NaN values (if appropriate for your use case)
        # group_data = group_data.dropna(subset=['value'])
        # Option 2: Fill NaN values with a specific value (e.g., fill with mean of the column)
        group_data['value'] = group_data['value'].fillna(group_data['value'].mean())
 
    # Fit the ARIMA model
    model = auto_arima(group_data['value'], seasonal=True, stepwise=True, suppress_warnings=True)
    
    next_minute_pred = model.predict(n_periods=10)
    # print("Predicted next minute value:", next_minute_pred)
    next_minute_pred=next_minute_pred.iloc[-1]
    # Calculate residuals (actual - predicted)
    group_data['predicted'] = model.predict_in_sample()  # Predict for the entire dataset (in-sample prediction)
    residuals = group_data['value'] - group_data['predicted']
    # print("Residuals calculated:", residuals.head())  # Print first few residuals for reference
 
    # Calculate Q1 (25th percentile), Q3 (75th percentile), and IQR
    Q1 = residuals.quantile(0.25)
    Q3 = residuals.quantile(0.75)
    IQR = Q3 - Q1
    # print('Q1:', Q1, 'Q3:', Q3, 'IQR:', IQR)
 
    # Define bounds for anomaly detection (1.5 * IQR for outliers)
    lower_bound = Q1 - 4.5 * IQR
    upper_bound = Q3 + 8.0 * IQR

    group_data['is_anomaly'] = ((residuals < lower_bound) | (residuals > upper_bound)).astype(int)
    group_data['Value_Predicted']=upper_bound 
    metricType=group_data["metricType"].iloc[0]
    host=group_data["host"].iloc[0]
    # print('next minute prediction Value',next_minute_pred)
    last_timestamp = last_1_day_data['timestamp'].iloc[-1]
    next_10_minutes_timestamps = pd.date_range(last_timestamp + pd.Timedelta(minutes=10), periods=1, freq='min')
    # print('Last time stamp verfication',next_10_minutes_timestamps)
    # next10min=next_10_minutes_timestamps.iloc[0]
    next_minute_prediction  = pd.DataFrame({
        'timestamp': next_10_minutes_timestamps,
        'predicted_value': next_minute_pred
    })
    # print('next_minute_prediction',next_minute_prediction)
    next_minute_prediction['server']=host
    next_minute_prediction['metric']=metricType
    # next_minute_prediction.to_sql(name='prediction_all_next_timestamp1', con=conn, if_exists='append', index=False)
    # print(next_minute_prediction)
    group_data['timestamp'] = pd.to_datetime(group_data['timestamp'], errors='coerce')
    group_data['timestamp'] = group_data['timestamp'].dt.tz_localize('UTC')
    group_data['timestamp'] = group_data['timestamp'].dt.tz_convert('Asia/Kolkata')
    latest_5_rows = group_data.sort_values(by='timestamp', ascending=False).head(5)
    latest_5_rows = latest_5_rows.drop_duplicates(subset=['host', 'timestamp', 'metricType'],keep='first')

    # print(len(latest_5_rows))
    # print(latest_5_rows)
    # latest_5_rows.to_sql(name='prediction_all_server_metrics3', con=conn, if_exists='append', index=False)  
    final=latest_5_rows
    final['FutureTimestamp']=next_minute_prediction['timestamp']
    final['Futurepredictedvalue']=next_minute_pred 
    # print(final.head(10))
    # print(len(final))
    
    final.to_sql(name='prediction_all_server_metrics_final', con=conn, if_exists='append', index=False)  

def PythonScript(group_data):    #python-script Data
    
    df=pd.DataFrame(group_data)
    # print('result df',group_data)
    df['timestamp'] = pd.to_datetime(group_data['timestamp'],unit='ms')
    # df= df[df['metricType'] == 'CPU Usage %']
    # df= df[df['metricType'] == 'readOps Usage %']
    
    result = df.groupby(['host','CI_Tpye', 'metricType']).apply(detect_anomalies).reset_index(drop=True)
    result = result.to_dict('records')
    # print(result)
    return result


resdf=PythonScript(group_data)
print('Completed')
# resdf=pd.DataFrame(resdf)
# print('result',resdf.head())

