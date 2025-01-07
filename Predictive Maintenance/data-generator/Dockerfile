# Use a base Python image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Upgrade pip
RUN pip install --upgrade pip

# Copy dependencies
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the data-generator script and dataset files
COPY data_generator.py .
COPY Train_Data_CSV.csv .
COPY Test_Data_CSV.csv .

# Run the data generator
CMD ["python", "data_generator.py"]
