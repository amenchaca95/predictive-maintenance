# Usa una imagen base de Python
FROM python:3.9

# Establecer el directorio de trabajo
WORKDIR /app

# Copiar los archivos del modelo y dependencias
COPY requirements.txt ./

# Instalar dependencias
RUN pip install --no-cache-dir -r requirements.txt

# Copiar los scripts del modelo al contenedor
COPY . .

# Definir el comando por defecto para ejecutar el modelo
CMD ["python", "train_predict.py"]
