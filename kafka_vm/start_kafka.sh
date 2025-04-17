# To be executed inside kafka_vm
# Start Kafka

# Navigate to the Kafka directory
cd kafka

KAFKA_EXTERNAL_IP=$(curl -s ifconfig.me)

# Update (or append) the listeners configuration. This ensures that Kafka binds to all interfaces and advertises the correct external address.
if grep -q "^listeners=" config/server.properties; then
    sed -i 's|^listeners=.*|listeners=PLAINTEXT://0.0.0.0:9092|' config/server.properties
else
    echo "listeners=PLAINTEXT://0.0.0.0:9092" >> config/server.properties
fi

# Update (or append) the advertised.listeners configuration. This ensures that Kafka binds to all interfaces and advertises the correct external address.
if grep -q "^advertised.listeners=" config/server.properties; then
    sed -i "s|^advertised.listeners=.*|advertised.listeners=PLAINTEXT://$KAFKA_EXTERNAL_IP:9092|" config/server.properties
else
    echo "advertised.listeners=PLAINTEXT://$KAFKA_EXTERNAL_IP:9092" >> config/server.properties
fi

bin/kafka-server-start.sh config/server.properties