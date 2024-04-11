# Use an official Python runtime as a parent image
FROM python:3.9-slim

ENV PYTHONUNBUFFERED 1

# Install necessary dependencies
RUN apt update
RUN apt install build-essential
RUN apt install pkg-config
RUN apt install libhdf5-dev
RUN apt install nano

# Set the working directory in the container
WORKDIR /app

# Copy the source code
COPY FL_server/src/ /app/src

# Set the working directory to /app/src/
WORKDIR /app/src

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Change working directory back to /app/
WORKDIR /app

#ENTRYPOINT ["python", "fl.kmeans.py"]
#ENTRYPOINT ["python", "LR.py"]
CMD ["tail", "-f", "/dev/null"]
