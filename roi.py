import sys
import cv2
import json
import pika
import base64
import numpy as np
import redis
import time
import os
import warnings
warnings.filterwarnings("ignore")
from config import rabbitmq, detection_queues, redis_server
from logging_code import setup_logging

logger = setup_logging("ROI")

# Folder to save detected frames
SAVE_DIR = "roi_detected_frames"
os.makedirs(SAVE_DIR, exist_ok=True)


class ROIConsumer:

    def __init__(self):
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

            self.channel.exchange_declare(
                exchange=rabbitmq["detection_exchange"],
                exchange_type="direct",
                durable=False
            )

            for q, rk in [
                ("vehicle_queue",    "vehicle"),
                ("fire_smoke_queue", "fire_smoke"),
                ("safety_queue",     "safety"),
                ("person_queue_out", "person_out")
            ]:
                self.channel.queue_declare(queue=q, durable=True)
                self.channel.queue_bind(
                    exchange=rabbitmq["detection_exchange"],
                    queue=q,
                    routing_key=rk
                )

            self.redis_db = redis.StrictRedis(
                host=redis_server["host"],
                port=redis_server["port"],
                db=redis_server["db"],
                decode_responses=False
            )

            self.roi_coords = {}

            #  Video writers for each camera
            self.video_writers = {}
            self.frame_size = (500, 400)

            logger.info("ROI Consumer Ready")

        except Exception as e:
            logger.error(f"Init Error: {e}")

    def get_roi(self, w, h):
        cx, cy = w // 2, h // 2
        rw, rh = int(w * 0.4), int(h * 0.4)
        return cx - rw // 2, cy - rh // 2, cx + rw // 2, cy + rh // 2

    def is_inside_roi(self, bbox, roi):
        x1, y1, x2, y2 = bbox
        rx1, ry1, rx2, ry2 = roi
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        return rx1 <= cx <= rx2 and ry1 <= cy <= ry2

    def save_to_redis(self, key, frame):
        try:
            _, buf = cv2.imencode(".jpg", frame)
            self.redis_db.rpush(key, base64.b64encode(buf).decode())
            logger.info(f"Saved frame to Redis key: {key}")
        except Exception as e:
            logger.error(f"Redis Save Error: {e}")

    def get_video_writer(self, cam_id, label):
        key = f"{cam_id}_{label}"
        if key not in self.video_writers:
            #  Create separate video file for each camera and label
            cam_folder = os.path.join(SAVE_DIR, cam_id)
            os.makedirs(cam_folder, exist_ok=True)
            video_path = os.path.join(cam_folder, f"{label}_roi.avi")
            fourcc = cv2.VideoWriter_fourcc(*"XVID")
            writer = cv2.VideoWriter(video_path, fourcc, 8.0, self.frame_size)
            self.video_writers[key] = writer
            logger.info(f"Created video file: {video_path}")
        return self.video_writers[key]

    def process_message(self, ch, method, properties, body):
        try:
            data = json.loads(body.decode())
            meta = data["meta"]
            cam_id = meta["cam_id"]
            detections = meta["detections"]

            frame = cv2.imdecode(
                np.frombuffer(base64.b64decode(data["frame_b64"]), np.uint8),
                cv2.IMREAD_COLOR
            )

            if frame is None:
                return

            h, w, _ = frame.shape

            if cam_id not in self.roi_coords:
                self.roi_coords[cam_id] = self.get_roi(w, h)

            rx1, ry1, rx2, ry2 = self.roi_coords[cam_id]

            # Draw ROI box
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 3)
            cv2.putText(frame, "ROI", (rx1, ry1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            roi_detected = False

            for det in detections:
                label = det["class"]
                conf  = det["confidence"]
                x1, y1, x2, y2 = map(int, det["bbox"])

                # Draw detection box
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"{label} {conf:.2f}", (x1, y1 - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

                inside = self.is_inside_roi(
                    (x1, y1, x2, y2),
                    (rx1, ry1, rx2, ry2)
                )

                color = (0, 255, 0) if inside else (0, 0, 255)
                cv2.putText(frame, f"ROI:{inside}", (x1, y2 + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                logger.info(f"CAM={cam_id} LABEL={label} INSIDE_ROI={inside}")

                if inside:
                    roi_detected = True

                    #  Save to Redis
                    cam_redis_map = {
                        "c1": {"Car": "c1_vehicle_frames",   "Bike": "c1_vehicle_frames"},
                        "c2": {"Helmet": "c2_safety_frames", "Jacket": "c2_safety_frames"},
                        "c3": {"Fire": "c3_fire_frames",     "Smoke": "c3_smoke_frames"},
                        "c4": {"Person": "c4_person_frames"}
                    }

                    if cam_id in cam_redis_map:
                        redis_key = cam_redis_map[cam_id].get(label)
                        if redis_key:
                            self.save_to_redis(redis_key, frame)

                    # Save frame to video file
                    resized = cv2.resize(frame, self.frame_size)
                    writer = self.get_video_writer(cam_id, label)
                    writer.write(resized)
                    logger.info(f"Frame written to video -> CAM={cam_id} LABEL={label}")

            # Show window only when ROI detected
            if roi_detected:
                cv2.imshow(f"ROI - {cam_id}", cv2.resize(frame, self.frame_size))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    self.release_writers()
                    cv2.destroyAllWindows()
                    sys.exit(0)

        except Exception as e:
            logger.error(f"Process Message Error: {e}")

    def release_writers(self):
        for writer in self.video_writers.values():
            writer.release()
        logger.info("All video writers released")

    def start(self):
        while True:
            try:
                for q in ["vehicle_queue", "fire_smoke_queue", "safety_queue", "person_queue_out"]:
                    self.channel.basic_consume(
                        queue=q,
                        on_message_callback=self.process_message,
                        auto_ack=True
                    )
                logger.info("Waiting for frames...")
                self.channel.start_consuming()

            except Exception as e:
                logger.error(f"Connection lost, reconnecting: {e}")
                time.sleep(2)
                self.__init__()


if __name__ == "__main__":
    ROIConsumer().start()