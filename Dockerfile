# Use official Python image as base
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements (if exists) and install dependencies
COPY req_for_docker/requirements.txt /app/requirements.txt 
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project files
COPY main.py /app/main.py

# Expose the application port
EXPOSE 8000 

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]