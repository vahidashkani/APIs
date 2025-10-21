# Use lightweight Python
FROM python:3.10

WORKDIR /app

# Copy requirements and install
COPY ./requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /app/requirements.txt

# Copy entire app (including model.onnx & models.yml)
COPY . /app

# Expose port
EXPOSE 80

# Run FastAPI with Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "80"]


