# Create a Kafka Topic

# --partitions 1 --replication-factor 1

#!/bin/bash

# Variables
KAFKA_BROKER="$(curl -s ifconfig.me):9092" # Kafka broker's IP
TOPIC_NAME=$1                      # First argument: Topic name
PARTITIONS=${2:-1}                 # Second argument (default = 1 partition)
REPLICATION_FACTOR=${3:-1}         # Third argument (default = 1 replica)

# Validate inputs
if [ -z "$TOPIC_NAME" ]; then
    echo "Usage: $0 <topic-name> [partitions] [replication-factor]"
    exit 1
fi

# Check if Kafka is reachable
echo "Checking Kafka broker at $KAFKA_BROKER..."
nc -zv $(echo $KAFKA_BROKER | cut -d':' -f1) 9092
if [ $? -ne 0 ]; then
    echo "Error: Unable to reach Kafka broker at $KAFKA_BROKER. Check broker status and network settings."
    exit 1
fi

# Create Kafka topic
echo "Creating Kafka topic: $TOPIC_NAME with $PARTITIONS partitions and $REPLICATION_FACTOR replication factor..."
cd kafka
bin/kafka-topics.sh --create --topic "$TOPIC_NAME" --bootstrap-server "$KAFKA_BROKER" --partitions "$PARTITIONS" --replication-factor "$REPLICATION_FACTOR"

# Verify topic creation
echo "Verifying topic creation..."
bin/kafka-topics.sh --list --bootstrap-server "$KAFKA_BROKER" | grep "$TOPIC_NAME"
if [ $? -eq 0 ]; then
    echo "Kafka topic '$TOPIC_NAME' created successfully!"
else
    echo "Error: Failed to create Kafka topic '$TOPIC_NAME'. Check Kafka logs."
    exit 1
fi