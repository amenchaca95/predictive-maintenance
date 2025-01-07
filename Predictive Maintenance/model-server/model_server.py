import paho.mqtt.client as mqtt
import json
import numpy as np
import joblib
from collections import deque
from influxdb_client import InfluxDBClient, Point, WritePrecision

# Configuration for MQTT
BROKER = "mosquitto"
MQTT_TOPIC = "sensors/air_filter_health"

# Configuration for InfluxDB
INFLUX_URL = "http://influxdb:8086"
INFLUX_TOKEN = "pO_3K_lcL2nkXjd58PG82oAH1Gq2T1fTwkuT-BnQs2W7Mf1G-bohvbTKOjcA-5ejT-ILYZBUb12_ZBDfTYA-0g=="
INFLUX_ORG = "UPPA"
INFLUX_BUCKET = "air_filter_data"

# Load trained model and scaler
print("Loading Linear Regression model and scaler...")
model = joblib.load("linear_regression_model.pkl")
scaler = joblib.load("scaler.pkl")
print("Model and scaler loaded successfully.")

# Buffer for storing last 20 inputs
sequence_length = 20
data_buffer = deque(maxlen=sequence_length)  # Keeps only the last 20 values

# Connect to InfluxDB
influx_client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
write_api = influx_client.write_api()

# Callback when an MQTT message is received
def on_message(client, userdata, message):
    global data_buffer

    # Parse incoming JSON message
    payload = json.loads(message.payload.decode())

    # Extract only the required feature(s) - here, 'Differential_pressure'
    new_data = [payload["Differential_pressure"]]

    # Append new data to the buffer
    data_buffer.append(new_data)

    # Only make predictions if we have 20 time steps
    if len(data_buffer) == sequence_length:
        input_data = np.array(data_buffer).reshape(1, -1)  # Reshape to (1, 20)
        input_data_scaled = scaler.transform(input_data)

        # Predict RUL
        rul_pred = model.predict(input_data_scaled)[0]

        print(f"Predicted RUL: {rul_pred}")

        # Send prediction to InfluxDB
        point = Point("rul_predictions") \
            .tag("sensor", "air_filter") \
            .field("Differential_pressure", float(payload["Differential_pressure"])) \
            .field("Predicted_RUL", float(rul_pred)) \
            .time(time=None, write_precision=WritePrecision.NS)

        write_api.write(bucket=INFLUX_BUCKET, org=INFLUX_ORG, record=point)

# Connect to MQTT broker
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.connect(BROKER, 1883, 60)
client.subscribe(MQTT_TOPIC)
client.on_message = on_message

# Keep the service running
client.loop_forever()


