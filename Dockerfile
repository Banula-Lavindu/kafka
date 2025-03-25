# Use a full Python image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy requirements.txt first
COPY requirements.txt .

# Install system dependencies required for OpenCV and other libraries
RUN apt-get update && apt-get install -y \
	build-essential \
	libssl-dev \
	libffi-dev \
	python3-dev \
	libgl1-mesa-glx \
	libglib2.0-0 \
	libsm6 \
	libxext6 \
	libxrender-dev

# Install Python dependencies
RUN pip install --upgrade pip
# Fix contourpy version before installation
RUN sed -i 's/contourpy==1.3.1/contourpy==1.3.0/g' requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the application files
COPY . .

# Expose Flask port
EXPOSE 5000

# Run Flask application
CMD ["python", "server.py"]
