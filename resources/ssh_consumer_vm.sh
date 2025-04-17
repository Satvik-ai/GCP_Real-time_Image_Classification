#!/bin/bash

# Variables
ZONE="us-west1-a"
VM_NAME="consumer-vm"

# SSH into the VM
gcloud compute ssh $VM_NAME --zone=$ZONE