# Imagen base
FROM python:3.10

# Evita archivos pyc y buffers
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Instalar dependencias del sistema
RUN apt-get update && \
    apt-get install -y python3-opencv && \
    apt-get clean

# Directorio de trabajo
WORKDIR /app

# Copiar proyecto
COPY . .

# Instalar dependencias de Python
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

# Ejecutar aplicación
CMD ["python", "app.py"]