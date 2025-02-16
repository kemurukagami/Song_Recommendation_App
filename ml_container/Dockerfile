FROM python:3.9-slim-bullseye

# Set working directory inside the container
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the ML model script into the container
COPY generate_model.py .

# Run the script automatically when the container starts
CMD ["python", "generate_model.py"]
