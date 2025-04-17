# To be executed inside producer_vm
# Install Java
sudo apt update
sudo apt install default-jdk -y
java -version
# Download & Extract Spark
wget https://downloads.apache.org/spark/spark-3.5.5/spark-3.5.5-bin-hadoop3.tgz
tar -xvzf spark-3.5.5-bin-hadoop3.tgz
mv spark-3.5.5-bin-hadoop3 spark
sudo apt update && sudo apt install netcat -y
sudo apt update && sudo apt install -y google-cloud-sdk python3 python3-pip scala
pip3 install google-cloud-storage kafka-python pyspark pandas tensorflow pathlib Pillow pyarrow