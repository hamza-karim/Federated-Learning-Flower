
# Use an official Python runtime as a parent image
FROM python:3.9-slim

ENV PYTHONUNBUFFERED 1

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container at /app
COPY src/ .

# Update package
RUN apt update
RUN apt install nano

RUN pip install --no-cache-dir -r requirements.txt

#ENTRYPOINT ["python", "fl.kmeans.py"]
#ENTRYPOINT ["python", "LR.py"]
CMD ["tail", "-f", "/dev/null"]
