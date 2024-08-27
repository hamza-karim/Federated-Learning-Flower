## Use an official Python runtime as a parent image
#FROM python:3.9-slim
#
#ENV PYTHONUNBUFFERED 1
#
## Set the working directory in the container
#WORKDIR /app
#
## Copy the source code
#COPY FL_server/src/ /app/src
#
## Set the working directory to /app/src/
#WORKDIR /app/src
#
### Update package
##RUN apt update
##RUN apt install nano
##RUN apt install -y pkg-config
##RUN apt install -y libhdf5-dev
#
## Update package lists and install necessary build tools and dependencies
#RUN apt-get update && apt-get install -y \
#    nano \
#    pkg-config \
#    libhdf5-dev \
#    gcc \
#    && rm -rf /var/lib/apt/lists/*
#
## Install Python dependencies
#RUN pip install --no-cache-dir -r requirements.txt
#
## Change working directory back to /app/
#WORKDIR /app
#
##ENTRYPOINT ["python", "server_RF.py"]
#CMD ["tail", "-f", "/dev/null"]

###################################################################################################

# Use the CUDA runtime container as the base image
#FROM nvcr.io/nvidia/l4t-cuda:12.2.12-devel
FROM nvcr.io/nvidia/l4t-ml:r32.5.0-py3

ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy the source code
COPY FL_server/src/ /app/src

# Set the working directory to /app/src/
WORKDIR /app/src

# Install necessary packages and pip
RUN apt-get update && apt-get install -y \
    nano \
    pkg-config \
    libhdf5-dev \
    gcc \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# Debug: Check if pip3 is available
RUN pip3 --version

# Debug: Check if requirements.txt is present
RUN ls -l /app/src/requirements.txt

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt || { echo 'Pip install failed'; exit 1; }

# Change working directory back to /app/
WORKDIR /app

# Keep the container running (replace with your actual command)
CMD ["tail", "-f", "/dev/null"]

