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

########################################(With GPU)###########################################################

#FROM nvcr.io/nvidia/l4t-ml:r35.2.1-py3
#Latest version
FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3

ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy the source code
COPY FL_server/src/ /app/src

# Set the working directory to /app/src/
WORKDIR /app/src

# Install necessary packages
RUN apt-get update && apt-get install -y \
    nano \
    pkg-config \
    libhdf5-dev \
    gcc \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*


#RUN pip3 install --no-cache-dir -r requirements.txt
RUN pip3 install --no-cache-dir -r requirements.txt

# Change working directory back to /app/
WORKDIR /app

# Keep the container running (replace with your actual command)
CMD ["tail", "-f", "/dev/null"]

