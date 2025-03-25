import cv2
import numpy as np
from pykafka import KafkaClient

# Kafka configuration
KAFKA_BROKER = "localhost:9092"
TOPIC = "new_cctv"

# Connect to Kafka
client = KafkaClient(hosts=KAFKA_BROKER)
topic = client.topics[TOPIC.encode("utf-8")]
consumer = topic.get_simple_consumer()

for message in consumer:
    if message is not None:
        # Decode the frame
        frame = np.frombuffer(message.value, dtype=np.uint8)
        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

        if frame is not None:
            cv2.imshow("Kafka Webcam Stream (Consumer)", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cv2.destroyAllWindows()
