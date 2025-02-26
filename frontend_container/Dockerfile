FROM python:3.9-slim-bullseye

WORKDIR /app

# Copy dependency file and install packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY templates/ templates/

# Copy your Flask application code
COPY recommender_app.py .

# Set environment variable so Flask knows which app to run
ENV FLASK_APP=recommender_app.py

# Expose the port Flask will run on inside the container
EXPOSE 52008

# Run Flask so it listens on all network interfaces
CMD ["flask", "run", "--host=0.0.0.0", "--port", "52008"]
