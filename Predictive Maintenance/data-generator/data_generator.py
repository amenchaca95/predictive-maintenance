import paho.mqtt.client as mqtt
import pandas as pd
import time
import json

# Conectar al broker MQTT
BROKER = "mosquitto"
MQTT_TOPIC = "sensors/air_filter_health"  # Se usa el nuevo topic
client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.connect(BROKER, 1883, 60)

# Cargar dataset
train_df = pd.read_csv("/app/Train_Data_CSV.csv")
test_df = pd.read_csv("/app/Test_Data_CSV.csv")

# Ordenar por tiempo
columns = ["Time", "Differential_pressure", "Flow_rate", "Dust_feed"]
train_df = train_df[columns].sort_values(by="Time")
test_df = test_df[columns].sort_values(by="Time")
simulated_data = pd.concat([train_df, test_df])

# Enviar datos secuenciales simulando sensores en tiempo real
for _, row in simulated_data.iterrows():
    data = {
        "Time": row["Time"],
        "Differential_pressure": row["Differential_pressure"],
        "Flow_rate": row["Flow_rate"],
        "Dust_feed": row["Dust_feed"]
    }
    
    client.publish(MQTT_TOPIC, json.dumps(data))
    print(f"Published: {data}")
    
    time.sleep(2)  # Simulación en tiempo real
