import pika
import json
import base64
import cv2
import numpy as np
import os
import sys
import time
import torch
from config import rabbitmq, queues
from logging_code import setup_logging
from redis_server import RedisServer  # ← Added

logger = setup_logging("DETECTION_SUBSCRIBER")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(BASE_DIR, "yolov7_detection"))

from models.experimental import attempt_load
from utils.general import non_max_suppression

COLORS = {
    "Car":    (255, 230, 78),
    "Bike":   (56, 255, 255),
    "Fire":   (255, 0, 255),
    "Smoke":  (150, 200, 128),
    "Helmet": (255, 155, 100),
    "Jacket": (155, 190, 255),
    "Person": (0, 255, 0)
}

CAMERA_CLASSES = {
    "c1": ["Car", "Bike"],
    "c2": ["Helmet", "Jacket"],
    "c3": ["Fire", "Smoke"],
    "c4": ["Person"]
}


class DetectionSubscriber:

    def __init__(self):
        try:
            self.connect_rabbitmq()
            self.redis = RedisServer()  # ← Added

            # Load YOLOv7 model
            weights = os.path.join(BASE_DIR, "best.pt")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

            with torch.no_grad():
                self.model = attempt_load(weights, map_location=self.device)
            self.model.eval()

            logger.info(f"YOLOv7 Loaded Successfully on {self.device}")
            logger.info(f"Classes: {self.model.names}")

        except Exception as e:
            logger.error(f"Init Error: {e}")

    def connect_rabbitmq(self):
        try:
            self.connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=rabbitmq["host"],
                    port=rabbitmq["port"],
                    heartbeat=600,
                    blocked_connection_timeout=300
                )
            )

            self.channel = self.connection.channel()

            # Main exchange
            self.channel.exchange_declare(
                exchange=rabbitmq["exchange"],
                exchange_type=rabbitmq["exchange_type"],
                durable=False
            )

            # Listen to camera queues (c1, c2, c3, c4)
            for cam in ["c1", "c2", "c3", "c4"]:
                queue_name = f"{cam}_queue"

                self.channel.queue_declare(
                    queue=queue_name,
                    durable=True
                )

                self.channel.queue_bind(
                    exchange=rabbitmq["exchange"],
                    queue=queue_name,
                    routing_key=cam
                )

            # Detection output exchange
            self.channel.exchange_declare(
                exchange=rabbitmq["detection_exchange"],
                exchange_type="direct",
                durable=False
            )

            # Output queues
            for q, rk in [
                ("vehicle_queue", "vehicle"),
                ("fire_smoke_queue", "fire_smoke"),
                ("safety_queue", "safety"),
                ("person_queue_out", "person_out")
            ]:
                self.channel.queue_declare(queue=q, durable=True)
                self.channel.queue_bind(
                    exchange=rabbitmq["detection_exchange"],
                    queue=q,
                    routing_key=rk
                )

            logger.info("RabbitMQ Connected Successfully")

        except Exception as e:
            logger.error(f"RabbitMQ Connect Error: {e}")

    def process_frame(self, frame, allowed=[]):
        try:
            original = frame.copy()
            detections = []

            img = cv2.cvtColor(cv2.resize(frame, (640, 640)), cv2.COLOR_BGR2RGB)
            tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
            tensor = tensor.to(self.device)

            with torch.no_grad():
                pred = non_max_suppression(self.model(tensor)[0], 0.25, 0.45)[0]

            if pred is not None:
                for *xyxy, conf, cls in pred:
                    x1, y1, x2, y2 = map(int, xyxy)
                    label = self.model.names[int(cls)]

                    if allowed and label not in allowed:
                        continue

                    color = COLORS.get(label, (0, 255, 0))
                    cv2.rectangle(original, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(original, f"{label} {conf:.2f}", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                    detections.append({
                        "class": label,
                        "confidence": float(conf),
                        "bbox": [x1, y1, x2, y2]
                    })
                    logger.info(f"Detected: {label} | Confidence: {conf:.2f}")

            return original, detections

        except Exception as e:
            logger.error(f"Process Frame Error: {e}")
            return frame, []

    def publish_detections(self, frame, detections, camera_name):
        try:
            if not detections:
                return

            _, buf = cv2.imencode(".jpg", frame)
            frame_b64 = base64.b64encode(buf).decode()

            routes = {
                "c1": (["Car", "Bike"],      "vehicle"),
                "c2": (["Helmet", "Jacket"], "safety"),
                "c3": (["Fire", "Smoke"],    "fire_smoke"),
                "c4": (["Person"],           "person_out")
            }

            if camera_name not in routes:
                return

            labels, routing_key = routes[camera_name]
            filtered = [d for d in detections if d["class"] in labels]

            if filtered:
                payload = {
                    "meta": {
                        "cam_id": camera_name,
                        "detections": filtered
                    },
                    "frame_b64": frame_b64
                }

                if not self.connection or self.connection.is_closed:
                    logger.info("Reconnecting RabbitMQ...")
                    self.connect_rabbitmq()

                self.channel.basic_publish(
                    exchange=rabbitmq["detection_exchange"],
                    routing_key=routing_key,
                    body=json.dumps(payload)
                )
                logger.info(f"Published to {routing_key} -> {[d['class'] for d in filtered]}")

        except Exception as e:
            logger.error(f"Publish Error: {e}")
            try:
                time.sleep(1)
                self.connect_rabbitmq()
                logger.info("Reconnected Successfully")
            except Exception as re:
                logger.error(f"Reconnect Error: {re}")

    def callback(self, ch, method, properties, body):
        try:
            data = json.loads(body)
            camera_name = data["camera_name"]

            frame = cv2.imdecode(
                np.frombuffer(base64.b64decode(data["frame"]), np.uint8),
                cv2.IMREAD_COLOR
            )

            if frame is None:
                return

            allowed = CAMERA_CLASSES.get(camera_name, [])
            detected_frame, detections = self.process_frame(frame, allowed)

            # ← Save detected frame to Redis (only if detections found)
            if detections:
                self.redis.save_frame_to_redis(camera_name, detected_frame)

            self.publish_detections(detected_frame, detections, camera_name)

            cv2.imshow(f"Detection - {camera_name}", cv2.resize(detected_frame, (500, 400)))
            if cv2.waitKey(1) == ord("q"):
                cv2.destroyAllWindows()
                sys.exit(0)

        except Exception as e:
            logger.error(f"Callback Error: {e}")

    def start(self):
        while True:
            try:
                # Consume all camera queues
                for cam in ["c1", "c2", "c3", "c4"]:
                    self.channel.basic_consume(
                        queue=f"{cam}_queue",
                        on_message_callback=self.callback,
                        auto_ack=True
                    )

                logger.info("Waiting for frames from c1, c2, c3, c4...")
                self.channel.start_consuming()

            except Exception as e:
                logger.error(f"Connection lost, reconnecting: {e}")
                time.sleep(2)
                self.__init__()


if __name__ == "__main__":
    DetectionSubscriber().start()