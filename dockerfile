# Use the official Python 3.9 image from the Docker Hub
FROM python:3.9-slim

# Set environment variables to ensure Python output is unbuffered and to avoid writing .pyc files
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install make and other necessary packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    make \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Create and set the working directory
WORKDIR /app

# Copy the requirements file into the container
COPY requirements.txt .

# Install the dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application code into the container
COPY . .

# Expose the port that the Flask app runs on
EXPOSE 5000

# Command to run the application (adjust if needed)
CMD ["make", "run-server"]