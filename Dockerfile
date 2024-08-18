# Base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy the requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install ffmpeg
RUN apt-get update && apt-get install -y ffmpeg && apt-get clean

# Copy the fonts and other necessary files
COPY fonts/ /app/fonts/
COPY main.py /app/

# Set environment variables if required
ENV FONT_PATH=/app/fonts/Oswald-Heavy.ttf

# Command to run the Flask app
CMD ["python", "main.py"]
