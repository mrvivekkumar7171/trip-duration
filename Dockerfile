# Use an official Python runtime as a parent image
FROM python:3.8-slim

# Set the working directory to /app
WORKDIR /app

# Copy the required files and directory into the container at /app
COPY app.py /app/app.py
COPY model.joblib /app/models/model.joblib
COPY src/ /app/src/
COPY requirements.txt /app/requirements.txt

# Install any needed packages specified in requirements.txt
RUN pip install -r requirements.txt

EXPOSE 8080

# Run app.py when the container launches
CMD ["python", "app.py"]