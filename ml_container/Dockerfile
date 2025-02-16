# Use a lightweight Python image
FROM python:3.9-slim-bullseye

# Set the working directory
WORKDIR /app

# Copy only the dependency file and install packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your model generation code into the container
COPY model_creation.py .

# Run the model generation script on container start
CMD ["python", "model_creation.py"]
