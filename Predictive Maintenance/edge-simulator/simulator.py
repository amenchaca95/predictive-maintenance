from influxdb_client import InfluxDBClient, Point, WritePrecision
import time
import random

# MQTT Setup
import paho.mqtt.client as mqtt
BROKER = "mosquitto"
MQTT_TOPIC = "sensors/machine_health"

# InfluxDB Setup
INFLUX_URL = "http://influxdb:8086"
INFLUX_TOKEN = "pO_3K_lcL2nkXjd58PG82oAH1Gq2T1fTwkuT-BnQs2W7Mf1G-bohvbTKOjcA-5ejT-ILYZBUb12_ZBDfTYA-0g=="
INFLUX_ORG = "UPPA"
INFLUX_BUCKET = "sensor_data"

client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
write_api = client.write_api()

# MQTT Publish Function
def publish_sensor_data():
    while True:
        vibration = round(random.uniform(0.5, 5.0), 2)
        temperature = round(random.uniform(50, 100), 2)
        data = {"machine_id": "crusher_1", "vibration": vibration, "temperature": temperature}

        # Publish to MQTT
        mqtt_client.publish(MQTT_TOPIC, str(data))

        # Write to InfluxDB
        point = Point("machine_health") \
            .tag("machine_id", "crusher_1") \
            .field("vibration", vibration) \
            .field("temperature", temperature)
        write_api.write(bucket=INFLUX_BUCKET, record=point)

        print(f"Published: {data}")
        time.sleep(5)

# Connect to MQTT
mqtt_client = mqtt.Client("edge_simulator")
mqtt_client.connect(BROKER, 1883, 60)
mqtt_client.loop_start()

# Start publishing sensor data
publish_sensor_data()
