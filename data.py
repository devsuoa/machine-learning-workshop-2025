import pandas as pd

df = pd.read_csv('food_time.csv')
df = df.drop(df.columns[14], axis=1)
df = df.drop(columns=['ID', 'Delivery_person_ID'])
df = df.dropna()

order = df['order_type'].unique()
order_map = {order[i]: i for i in range(len(order))}
df['order_type'] = df['order_type'].map(order_map)

vehicle = df['vehicle_type'].unique()
vehicle_map = {vehicle[i]: i for i in range(len(vehicle))}
df['vehicle_type'] = df['vehicle_type'].map(vehicle_map)

weather = df['weather'].unique()
# print(weather)
weather_map = {weather[i]: i for i in range(len(weather))}
df['weather'] = df['weather'].map(weather_map)

# traffic = df['traffic'].unique()
traffic = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
traffic_map = {traffic[i]: i for i in range(len(traffic))}
df['traffic'] = df['traffic'].map(traffic_map)

df.drop(columns=['source_lat', 'source_long', 'dest_lat', 'dest_long'], inplace=True)

df.to_csv('data.csv', index=False)