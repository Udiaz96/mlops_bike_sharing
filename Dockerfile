# Use Python 3.12 slim image
FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the bike_sharing package
COPY bike_sharing/ ./bike_sharing/

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["uvicorn", "bike_sharing.api_server:app", "--host", "0.0.0.0", "--port", "8000"]
