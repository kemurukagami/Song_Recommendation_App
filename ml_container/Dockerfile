# Use a lightweight Python image
FROM python:3.9-slim-bullseye

# Set the working directory
WORKDIR /app

# Copy only the dependency file and install packages
COPY requirements_ml.txt .
RUN pip install --no-cache-dir -r requirements_ml.txt

# Copy your model generation code into the container
COPY generate_model.py .

# Run the model generation script on container start
CMD ["python", "model_creation.py"]
