FROM python:3.10

WORKDIR /app

COPY . .

# Install a compatible version of paho-mqtt (1.6.1)
RUN pip install paho-mqtt==1.6.1
RUN pip install influxdb-client

CMD ["python", "simulator.py"]
