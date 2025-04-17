from pyspark.sql import SparkSession
from pyspark.sql.functions import col, pandas_udf, regexp_extract
from pyspark.sql.types import StructType, StructField, IntegerType
import tensorflow as tf
import pathlib
from PIL import Image
import pandas as pd
import io
from kafka import KafkaProducer
import json
import time
import base64

# Initialize Spark Session
spark = SparkSession.builder \
    .appName("ImageProducer") \
    .getOrCreate()

data_dir = tf.keras.utils.get_file(origin='https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz',fname='flower_photos', untar=True)

images = spark.read.format("binaryFile").option("recursiveFileLookup", "true").option("pathGlobFilter", "*.jpg").load(data_dir)

def extract_label(path_col):
    """Extract label from file path using built-in SQL functions."""
    return regexp_extract(path_col, "flower_photos/([^/]+)", 1)

# Define the schema with nullable=True to match what the UDF returns
size_schema = StructType([
    StructField("width", IntegerType(), True),
    StructField("height", IntegerType(), True)
])

@pandas_udf(size_schema)
def extract_size_udf(content_series):
    """Extract image dimensions from content bytes."""
    widths = []
    heights = []
    
    for content in content_series:
        try:
            image = Image.open(io.BytesIO(content))
            width, height = image.size
            widths.append(width)
            heights.append(height)
        except Exception as e:
            # Handle corrupt images or other errors
            widths.append(None)
            heights.append(None)
    
    return pd.DataFrame({'width': widths, 'height': heights})

# Process the images
df = images.select(
    col("path"),
    col("modificationTime"),
    extract_label(col("path")).alias("label"),
    extract_size_udf(col("content")).alias("size"),
    col("content"))

# Flatten the struct column for easier viewing
df = df.select("path", "modificationTime", "label", "size.*", "content")

# Show schema and sample data
df.printSchema()
df.show(5, truncate=True)

# Show all columns except content
columns_to_show = [c for c in df.columns if c != 'content']
df.select(*columns_to_show).show(5, truncate=False)

KAFKA_TOPIC = "input-topic"
KAFKA_BOOTSTRAP_SERVERS = "34.30.24.136:9092"

producer = KafkaProducer(bootstrap_servers=KAFKA_BOOTSTRAP_SERVERS, value_serializer=lambda v: json.dumps(v).encode('utf-8'),max_request_size=10485760)

# Function to send a row to Kafka
def send_to_kafka(row, producer, topic):

    # Convert binary content to base64 encoded string
    content_base64 = base64.b64encode(row["content"]).decode('utf-8')

    # Convert row to dictionary
    row_dict = {
        "path": row["path"],
        "modificationTime": str(row["modificationTime"]),
        "label": row["label"],
        "width": row["width"],
        "height": row["height"],
        "content": content_base64
    }
    
    # Send to Kafka
    producer.send(topic, row_dict)
    producer.flush()
    print(f"Sent row with path {row['path']} to Kafka topic {topic}")

rows = df.collect()
for row in rows:
    send_to_kafka(row, producer, KAFKA_TOPIC)
    time.sleep(1)  # 1 second delay

print(f"Finished sending {len(rows)} rows to Kafka topic {KAFKA_TOPIC}")

# Stop Spark Session
spark.stop()