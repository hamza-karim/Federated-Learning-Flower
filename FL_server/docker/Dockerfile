#Latest version
FROM nvcr.io/nvidia/l4t-ml:r36.2.0-py3
ENV PYTHONUNBUFFERED=1

# Set the working directory in the container
WORKDIR /app

# Copy the source code
COPY FL_server/src/ /app/src

# Set the working directory to /app/src/
WORKDIR /app/src

# # tensorflow GPU core dump problem
# ENV TF_FORCE_GPU_ALLOW_GROWTH=true
# ENV TF_GPU_ALLOCATOR=cuda_malloc_async
# ENV OMP_NUM_THREADS=1

RUN pip3 install --no-cache-dir -r requirements.txt

# Change working directory back to /app/
WORKDIR /app

# Keep the container running
CMD ["tail", "-f", "/dev/null"]

