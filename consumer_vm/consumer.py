from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf, from_json, udf
from pyspark.sql.types import StringType, StructType, StructField, BinaryType, IntegerType
from torchvision import models

KAFKA_TOPIC = "input-topic"
KAFKA_BOOTSTRAP_SERVERS = "34.172.110.143:9092"

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("KafkaSparkStreamingConsumer") \
    .getOrCreate()

# Set log level to reduce verbosity
spark.sparkContext.setLogLevel("WARN")

# Print debug information
print(f"Connecting to Kafka broker at {KAFKA_BOOTSTRAP_SERVERS}")
print(f"Subscribing to topic {KAFKA_TOPIC}")

# Define schema for incoming JSON data
schema = StructType([
    StructField("path", StringType(), True),
    StructField("modificationTime", StringType(), True),
    StructField("label", StringType(), True),
    StructField("width", IntegerType(), True),
    StructField("height", IntegerType(), True),
    StructField("content", StringType(), True)
])

# Read from Kafka
df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
    .option("subscribe", KAFKA_TOPIC) \
    .option("startingOffsets", "latest") \
    .load()

# Deserialize JSON messages and keep the timestamp field
df = df.selectExpr("CAST(value AS STRING) AS value", "timestamp")
df = df.withColumn("parsed_data", from_json(col("value"), schema)) \
       .select("timestamp", 
               "parsed_data.path", 
               "parsed_data.modificationTime", 
               "parsed_data.label", 
               "parsed_data.width", 
               "parsed_data.height", 
               "parsed_data.content")

model_broadcast = spark.sparkContext.broadcast(models.mobilenet_v2(pretrained=True))

# Regular Python UDF for single-image processing
@udf(StringType())
def predict(content):
    import base64
    import io
    from PIL import Image
    import torch
    from torchvision import transforms
    import json
    import requests

    if not content or content.strip() == "":
        return "Error: Empty content"
    
    try:
        # Load ImageNet labels (1000 classes)
        url = "https://storage.googleapis.com/download.tensorflow.org/data/imagenet_class_index.json"
        imagenet_classes = requests.get(url).json()
        idx_to_label = {int(k): v[1] for k, v in imagenet_classes.items()}  # {index: label}

        # Get model from broadcast variable
        model = model_broadcast.value
        model.eval()
        
        # Decode base64 content
        image_bytes = base64.b64decode(content)
        
        # Process image
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        # Run prediction
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted_class = output.max(1)
            class_index = predicted_class.item()

            # Convert index to label
            class_label = idx_to_label.get(class_index, "Unknown")
            return class_label
    except Exception as e:
        return f"Error: {str(e)}"
    
# Apply the UDF to the content column
df = df.withColumn("prediction", predict(col("content")))

# Select the columns to display
display_df = df.select("path", "label", "prediction")

# Write predictions to console
query = display_df.writeStream \
    .outputMode("append") \
    .format("console") \
    .option("truncate", "false") \
    .start()

query.awaitTermination()