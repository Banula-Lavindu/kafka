import cv2
import numpy as np
import threading
from pykafka import KafkaClient
import time

# Kafka Configuration
KAFKA_BROKER = "35.226.126.108:9092"
TOPIC = "cctv"

# List of RTSP Camera URLs
CCTV_CAMERAS = {
    "cam1": "rtsp://admin:BPLVBI@192.168.24.153:554/h264/ch01/sub/av_stream",
    "cam2": "rtsp://admin:KYABCC@192.168.24.144:554/Streaming/Channels/102",
    "cam3": "rtsp://admin:FBBWWA@192.168.24.122:554/Streaming/Channels/102",
    "cam4": "rtsp://admin:KYABCC@192.168.24.144:554/Streaming/Channels/202",
    "cam5": "rtsp://admin:FBBWWA@192.168.24.122:554/Streaming/Channels/202"
}

# Connect to Kafka
client = KafkaClient(hosts=KAFKA_BROKER)
topic = client.topics[TOPIC.encode("utf-8")]
# Configure producer to only handle new messages
producer = topic.get_sync_producer(
    delivery_reports=False,  # Disable delivery reports for performance
    linger_ms=10,  # Small batching delay for better throughput
    max_request_size=1000000,  # Set max request size to 1MB
    compression=0,  # No compression (0), can use 1 for GZIP if needed
)

# Stop flag for clean shutdown
running = True

# Function to handle streaming for each CCTV
def stream_cctv(cam_name, rtsp_url):
    global running
    frame_skip = 2  # Process every 3rd frame
    frame_count = 0
    reconnect_delay = 5
    max_retry_delay = 30

    while running:
        try:
            # Set OpenCV to use TCP for RTSP (more reliable than UDP)
            os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
            
            # Create capture with specific options
            cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
            
            # Set additional parameters for stability
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)  # Keep a small buffer
            
            if not cap.isOpened():
                print(f"⚠️ Error: Cannot open RTSP stream for {cam_name}. Retrying in {reconnect_delay}s...")
                time.sleep(reconnect_delay)
                reconnect_delay = min(reconnect_delay * 1.5, max_retry_delay)  # Exponential backoff
                continue

            print(f"✅ Streaming {cam_name} to Kafka...")
            reconnect_delay = 5  # Reset delay on successful connection
            consecutive_errors = 0

            while running:
                ret, frame = cap.read()
                if not ret:
                    consecutive_errors += 1
                    if consecutive_errors >= 3:
                        print(f"⚠️ Error: Failed to read frame from {cam_name}. Reconnecting...")
                        break  # Restart RTSP connection
                    continue
                
                consecutive_errors = 0
                frame_count += 1
                
                # Skip frames to reduce load
                if frame_count % (frame_skip + 1) != 0:
                    continue
                    
                # Resize frame to reduce message size
                frame = cv2.resize(frame, (640, 360))

                # Encode frame to JPEG with compression
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 50])

                # Ensure message size is within Kafka limits (below 900 KB)
                if len(buffer) > 900000:
                    print(f"⚠️ Skipping large frame from {cam_name} ({len(buffer)} bytes)")
                    continue

                # Send frame to Kafka
                try:
                    producer.produce(f"{cam_name}".encode() + b":" + buffer.tobytes())
                except Exception as e:
                    print(f"⚠️ Kafka Error: {e}")

        except Exception as e:
            print(f"⚠️ Unexpected error with {cam_name}: {str(e)}")
        
        finally:
            if 'cap' in locals() and cap is not None:
                cap.release()
            
            time.sleep(reconnect_delay)  # Wait before retrying RTSP connection

    print(f"🛑 Stopped streaming {cam_name}")


# Add missing import
import os

# Create and start threads for each CCTV camera
threads = []
for cam_name, rtsp_url in CCTV_CAMERAS.items():
    t = threading.Thread(target=stream_cctv, args=(cam_name, rtsp_url), daemon=True)
    t.start()
    threads.append(t)

# Wait for all threads to complete
try:
    for t in threads:
        t.join()
except KeyboardInterrupt:
    print("\n🛑 Stopping all streams...")
    running = False
    time.sleep(2)  # Allow threads to stop gracefully