# Use the official Python image
FROM python:3.9-slim

# Define port argument and environment variable
ARG PORT=8080
ENV PORT=${PORT}

# Set working directory
WORKDIR /app

# Copy requirements
COPY requirements.txt ./

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy model artifacts
COPY models ./models

# Copy app files
COPY app ./app

# Expose port for Cloud Run
EXPOSE ${PORT}

# Command to run app using the PORT environment variable
CMD uvicorn app.main:app --host 0.0.0.0 --port $PORT 