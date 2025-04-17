## Task  
**Convert the batch image classification use case into a real-time execution model using Spark Streaming.**  
Original use case: [Batch Image Classification Use Case](https://drive.google.com/file/d/1BufNhnDKvuLA0Vd59pdPK8bCbTg1JZu8/view?usp=sharing)

## Approach
- Deployed a cloud-based real-time image classification pipeline using Google Cloud VMs.
- Provisioned and configured three VMs: Kafka broker, image producer, and Spark Streaming consumer.
- Implemented a producer to stream image data to a kafka topic one image at a time.
- Built a Spark Streaming consumer to read image data in real-time from the kafka topic and classify it using a pre-trained model from PyTorch’s torchvision.

## Report  
- [View the PDF](/21f1000344-IBD-GA9.pdf)

## Video Presentation  
[![Watch the video](images/video-thumb.png)](https://drive.google.com/file/d/1sr2ER-EDPXNodzyy21dAEex2GByoukUq/view?usp=sharing)