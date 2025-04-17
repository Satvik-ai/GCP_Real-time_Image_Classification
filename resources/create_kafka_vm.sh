# This script provisions a VM for running Kafka server and sets up a firewall rule 
# to allow incoming traffic on port 9092 only from the resources serving as consumer and producer:

#!/bin/bash

# Variables
VM_NAME="kafka-server-vm"
PROJECT_ID="eminent-crane-448810-s3"
ZONE="us-central1-a"
MACHINE_TYPE="c4-standard-4"
IMAGE_FAMILY="debian-11"
IMAGE_PROJECT="debian-cloud"

# Step 1: Create the Kafka server VM instance
gcloud compute instances create $VM_NAME \
    --project=$PROJECT_ID \
    --zone=$ZONE \
    --machine-type=$MACHINE_TYPE \
    --image-family=$IMAGE_FAMILY \
    --image-project=$IMAGE_PROJECT \
    --boot-disk-size=50GB \
    --scopes=storage-full,cloud-platform \
    --tags=kafka-server

# Step 2: Open required ports
gcloud compute firewall-rules create kafka-port --allow tcp:9092 --target-tags kafka-server --quiet