# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory
WORKDIR /app

# Copy only the requirements file to the working directory
COPY docker/requirements.txt /app/requirements.txt

# Install only the required dependencies from requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . /app

# Expose the port for the Streamlit app
EXPOSE 8501

# Run Streamlit app when the container launches
CMD ["streamlit", "run", "src/app_front.py", "--server.port=8501", "--server.address=0.0.0.0"]

