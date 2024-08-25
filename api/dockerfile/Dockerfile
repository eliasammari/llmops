# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy only the requirements file to the working directory
COPY dockerfile/requirements.txt /app/requirements.txt

# Install only the required dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . /app

# Expose the port for the Flask server
EXPOSE 5000

# Set the environment variable for the Flask app
ENV FLASK_APP=src/app.py

# Run the Flask app when the container launches
CMD ["flask", "run", "--host=0.0.0.0", "--port=5000"]
