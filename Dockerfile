# Use official Python image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

# Set work directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt /app/
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy project
COPY . /app/

# Expose port (Flask default is 5000)
EXPOSE 5000

# Set environment variables for Flask
ENV FLASK_APP=api_server.py
ENV FLASK_ENV=production

# Run with Gunicorn for production
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:5000", "api_server:app"]
