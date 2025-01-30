#!/bin/bash

# Pull the auto-sklearn Docker image
echo "Pulling the auto-sklearn Docker image..."
docker pull mfeurer/auto-sklearn:master

# Get the container ID of the running container
echo "Getting the container ID..."
container_id=$(docker ps -q -l)

# Copy the Python script and data files to the container
echo "Copying files to the container..."
docker cp AutoSklearn.py $container_id:/AutoSklearn.py
docker cp airfoil_noise_data.csv $container_id:/airfoil_noise_data.csv
docker cp CongressionalVotingID.shuf.lrn.csv $container_id:/CongressionalVotingID.shuf.lrn.csv
docker cp abalone.csv $container_id:/abalone.csv

# Install Python 3 in the container (if not already installed)
echo "Checking Python installation in the container..."
docker exec -it $container_id bash -c "python3 --version || apt-get update && apt-get install -y python3"

# Run the Python script in the container
echo "Running the Python script in the container..."
docker exec -it $container_id bash -c "python3 /AutoSklearn.py"