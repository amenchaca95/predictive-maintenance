# Use a base Python image
FROM python:3.9

# Set working directory
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Copy dependencies
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model server script and trained models
COPY model_server.py .
COPY linear_regression_model.pkl .
COPY scaler.pkl .

# Run the server
CMD ["python", "model_server.py"]



