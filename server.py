from flask import Flask, Response, render_template, jsonify, request, send_file
import cv2
import numpy as np
from pykafka import KafkaClient, exceptions
import threading
from ultralytics import YOLO  # Import YOLO
from flask_cors import CORS  # Import CORS for cross-origin requests
import json
import os
import time
import io
import socket
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Configure CORS more permissively

# Kafka Configuration - use external IP if server is on GCP
# Make sure this IP is accessible from both your local producer and the GCP instance
KAFKA_BROKER = "10.128.0.7:9092"
TOPIC = "cctv"
KAFKA_CONNECT_RETRY_INTERVAL = 5  # seconds

# Load YOLO model
try:
    model = YOLO('yolov8n.pt')  # Load pretrained YOLOv8 model (nano version)
    logger.info("YOLO model loaded successfully")
except Exception as e:
    logger.error(f"Error loading YOLO model: {e}")
    model = None

# Function to create a Kafka consumer
def get_kafka_consumer():
    """Create and return a Kafka consumer with error handling."""
    retry_count = 0
    max_retries = 5
    
    while retry_count < max_retries:
        try:
            logger.info(f"Attempting to connect to Kafka broker at {KAFKA_BROKER}")
            client = KafkaClient(hosts=KAFKA_BROKER, socket_timeout_ms=10000)
            topic = client.topics[TOPIC.encode("utf-8")]
            consumer = topic.get_simple_consumer(
                consumer_timeout_ms=5000,  # 5 seconds timeout
                auto_commit_enable=True,
                reset_offset_on_start=True,
                consumer_group=f'cctv-server-{socket.gethostname()}',  # Unique consumer group
                fetch_min_bytes=1024,  # Minimum amount of data to fetch
                fetch_wait_max_ms=2000  # Max time to wait for minimum bytes
            )
            logger.info("Successfully connected to Kafka")
            return consumer
        except exceptions.KafkaException as e:
            retry_count += 1
            logger.error(f"Error connecting to Kafka (attempt {retry_count}/{max_retries}): {e}")
            time.sleep(KAFKA_CONNECT_RETRY_INTERVAL)
    
    logger.error(f"Failed to connect to Kafka after {max_retries} attempts")
    return None

# Initialize Kafka consumer
consumer = get_kafka_consumer()

# Dictionary to store latest frames for each camera
latest_frames = {}
# Dictionary to store detection results
detection_results = {}
# Dictionary to store camera metadata
camera_info = {
    "cam1": {
        "id": "cam1",
        "name": "Front Door",
        "location": "Main Entrance",
        "isOnline": False,
        "lastUpdated": None
    },
    "cam2": {
        "id": "cam2",
        "name": "Back Door",
        "location": "Rear Entrance",
        "isOnline": False,
        "lastUpdated": None
    },
    "cam3": {
        "id": "cam3",
        "name": "Side Entrance",
        "location": "Side Entrance",
        "isOnline": False,
        "lastUpdated": None
    }
}

def process_frame_with_yolo(frame, cam_name):
    """Process frame with YOLO detection and return annotated frame"""
    if model is None:
        return frame
    
    # Run YOLO detection
    results = model(frame)
    
    # Store detection results
    detection_results[cam_name] = results[0]
    
    # Draw detection results on frame
    annotated_frame = results[0].plot()
    
    return annotated_frame

def consume_frames():
    """Continuously consume frames from Kafka and store them in a dictionary."""
    global consumer
    last_reconnect_attempt = 0
    reconnect_threshold = 10  # seconds

    while True:
        if consumer is None:
            current_time = time.time()
            if current_time - last_reconnect_attempt > reconnect_threshold:
                logger.info("Attempting to reconnect to Kafka...")
                consumer = get_kafka_consumer()
                last_reconnect_attempt = current_time
            else:
                time.sleep(1)  # Don't hammer reconnection attempts
                continue  

        try:
            # Mark all cameras as offline initially
            current_time = time.time()
            for cam_id in camera_info:
                if cam_id in latest_frames and current_time - camera_info[cam_id].get("lastSeen", 0) > 10:
                    if camera_info[cam_id]["isOnline"]:
                        logger.info(f"Camera {cam_id} is now offline")
                        camera_info[cam_id]["isOnline"] = False

            # Poll for messages
            message_count = 0
            for message in consumer:
                message_count += 1
                if message is not None:
                    try:
                        # Extract camera name and frame data
                        cam_name, frame_data = message.value.split(b":", 1)
                        cam_name = cam_name.decode()

                        # Decode frame
                        frame = np.frombuffer(frame_data, dtype=np.uint8)
                        frame = cv2.imdecode(frame, cv2.IMREAD_COLOR)

                        if frame is not None:
                            # Apply YOLO object detection
                            processed_frame = process_frame_with_yolo(frame, cam_name)
                            latest_frames[cam_name] = processed_frame
                            
                            # Update camera status
                            if cam_name in camera_info:
                                current_time = time.time()
                                previous_status = camera_info[cam_name]["isOnline"]
                                camera_info[cam_name]["isOnline"] = True
                                camera_info[cam_name]["lastSeen"] = current_time
                                camera_info[cam_name]["lastUpdated"] = json.dumps(
                                    {"$date": int(current_time * 1000)}
                                )
                                
                                if not previous_status:
                                    logger.info(f"Camera {cam_name} is now online")
                                    
                    except Exception as e:
                        logger.exception(f"Error processing frame: {e}")
                
                # Break after processing some messages to check for offline cameras
                if message_count > 10:
                    break
                    
            # If no messages received, sleep a bit to avoid CPU spin
            if message_count == 0:
                time.sleep(0.5)
                
        except exceptions.ConsumerStoppedException:
            logger.warning("Consumer stopped. Attempting to reconnect...")
            consumer = None
            time.sleep(1)
        except Exception as e:
            logger.exception(f"Error in consumer loop: {e}")
            consumer = None  # Reset consumer on error
            time.sleep(2)  # Wait before retry

# Start Kafka frame consumer in a separate thread
threading.Thread(target=consume_frames, daemon=True).start()

def generate_mjpeg(cam_name):
    """Generate MJPEG stream for a specific camera."""
    while True:
        if cam_name in latest_frames:
            # Encode the frame as JPEG
            _, jpeg = cv2.imencode('.jpg', latest_frames[cam_name], [cv2.IMWRITE_JPEG_QUALITY, 70])
            frame_bytes = jpeg.tobytes()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n\r\n' + 
                   frame_bytes + b'\r\n')
        else:
            # If no frame available, yield offline image
            with open('static/offline.jpg', 'rb') as f:
                offline_bytes = f.read()
            
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(offline_bytes)).encode() + b'\r\n\r\n' + 
                   offline_bytes + b'\r\n')
            
        # Rate limit the stream
        time.sleep(0.1)

@app.route('/')
def index():
    """Render the HTML page with multiple CCTV feeds."""
    return render_template('index.html', cameras=latest_frames.keys())

@app.route('/video_feed/<cam_name>')
def video_feed(cam_name):
    """Stream video feed for a specific camera."""
    return Response(
        generate_mjpeg(cam_name), 
        mimetype='multipart/x-mixed-replace; boundary=frame'
    )
    
@app.route('/frame/<cam_name>')
def get_frame(cam_name):
    """Get a single frame for a specific camera."""
    if cam_name in latest_frames:
        _, jpeg = cv2.imencode('.jpg', latest_frames[cam_name], [cv2.IMWRITE_JPEG_QUALITY, 80])
        return Response(jpeg.tobytes(), mimetype='image/jpeg')
    else:
        return send_file('static/offline.jpg', mimetype='image/jpeg')

@app.route('/detections/<cam_name>')
def get_detections(cam_name):
    """Get detection results for a specific camera in JSON format."""
    if cam_name in detection_results:
        result = detection_results[cam_name]
        detections = []
        
        if hasattr(result, 'boxes') and result.boxes is not None:
            for box in result.boxes:
                detection = {
                    'class': int(box.cls.item()),
                    'class_name': model.names[int(box.cls.item())],
                    'confidence': float(box.conf.item()),
                    'bbox': box.xyxy.tolist()[0]
                }
                detections.append(detection)
        
        return jsonify(detections)
    
    return jsonify([])

# API endpoint to get all cameras
@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    """Get all available cameras with status information."""
    camera_list = []
    for cam_id, info in camera_info.items():
        is_online = cam_id in latest_frames and info["isOnline"]
        camera_data = {
            "id": cam_id,
            "name": info["name"],
            "location": info["location"],
            "isOnline": is_online,
            "lastUpdated": info["lastUpdated"],
            "imageUrl": f"/thumbnail/{cam_id}" if is_online else "/static/offline.jpg"
        }
        camera_list.append(camera_data)
    
    return jsonify(camera_list)

# API endpoint to get specific camera information
@app.route('/api/cameras/<cam_id>', methods=['GET'])
def get_camera(cam_id):
    """Get information for a specific camera."""
    if cam_id in camera_info:
        is_online = cam_id in latest_frames and camera_info[cam_id]["isOnline"]
        camera_data = {
            "id": cam_id,
            "name": camera_info[cam_id]["name"],
            "location": camera_info[cam_id]["location"],
            "isOnline": is_online,
            "lastUpdated": camera_info[cam_id]["lastUpdated"],
            "streamUrl": f"/frame/{cam_id}" if is_online else None
        }
        return jsonify(camera_data)
    
    return jsonify({"error": "Camera not found"}), 404

# API endpoint to get thumbnail for a camera
@app.route('/thumbnail/<cam_id>')
def get_thumbnail(cam_id):
    """Get a thumbnail image for a specific camera."""
    if cam_id in latest_frames:
        _, jpeg = cv2.imencode('.jpg', latest_frames[cam_id], [cv2.IMWRITE_JPEG_QUALITY, 70])
        return Response(jpeg.tobytes(), mimetype='image/jpeg', headers={
            'Cache-Control': 'no-store, no-cache, must-revalidate, max-age=0',
            'Expires': '0'
        })
    else:
        return send_file('static/offline.jpg', mimetype='image/jpeg', as_attachment=False)

# Create static folder for offline image if it doesn't exist
if not os.path.exists('static'):
    os.makedirs('static')
    
# Create an offline image if it doesn't exist
if not os.path.exists('static/offline.jpg'):
    offline_img = np.zeros((360, 640, 3), dtype=np.uint8)
    cv2.putText(offline_img, "Camera Offline", (150, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imwrite('static/offline.jpg', offline_img)

if __name__ == "__main__":
    # Get the local IP for better network visibility
    def get_local_ip():
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
            s.close()
            return local_ip
        except:
            return "127.0.0.1"
    
    local_ip = get_local_ip()
    logger.info(f"Server running at: http://{local_ip}:5000")
    logger.info(f"Connected to Kafka broker at: {KAFKA_BROKER}")
    
    app.run(host="0.0.0.0", port=5000, debug=True, threaded=True)

