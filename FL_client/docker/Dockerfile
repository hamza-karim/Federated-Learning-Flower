## Use an official Python runtime as a parent image
#FROM python:3.9-slim
#
#ENV PYTHONUNBUFFERED 1
#
## Set the working directory in the container
#WORKDIR /app
#
## Copy the source code
#COPY FL_client/src/ /app/src
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
##ENTRYPOINT ["python", "fl.kmeans.py"]
##ENTRYPOINT ["python", "rf.py"]
#CMD ["tail", "-f", "/dev/null"]

####################################(With GPU)############################################################

#FROM nvcr.io/nvidia/l4t-ml:r32.7.1-py3
#FROM python:3.9-slim
#
#ENV PYTHONUNBUFFERED=1
#
## Set the working directory in the container
#WORKDIR /app
#
## Copy the source code
#COPY FL_client/src/ /app/src
#
## Set the working directory to /app/src/
#WORKDIR /app/src
#
## Install necessary packages and pip
#RUN apt-get update && apt-get install -y \
#    nano \
#    pkg-config \
#    libhdf5-dev \
#    gcc \
#    #python3-pip \
#    && rm -rf /var/lib/apt/lists/*
#
## Install Python dependencies
##RUN pip3 install --no-cache-dir -r requirements.txt || { echo 'installation failed'; exit 1; }
#RUN pip install --no-cache-dir -r requirements.txt
#
## Change working directory back to /app/
#WORKDIR /app
#
## Keep the container running (replace with your actual command)
#CMD ["tail", "-f", "/dev/null"]


#######################################################################################################

# Stage 1: Use the NVIDIA image to install system packages
FROM nvcr.io/nvidia/l4t-ml:r32.7.1-py3 AS base

# Install additional system packages needed for Python environment
RUN apt-get update && apt-get install -y \
    nano \
    pkg-config \
    libhdf5-dev \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Use Python 3.9-slim as the final image
FROM python:3.9-slim AS final

# Set environment variables
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy necessary files and directories from the base image
COPY --from=base /usr /usr
COPY --from=base /lib /lib
COPY --from=base /etc /etc

# Copy the source code into the container
COPY FL_client/src/ /app/src

# Set the working directory to /app/src/
WORKDIR /app/src

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Change working directory back to /app/
WORKDIR /app

# Keep the container running (replace with your actual command)
CMD ["tail", "-f", "/dev/null"]

