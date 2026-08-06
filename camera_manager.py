import cv2
import base64
import pika
import json
import time
import sys
from config import rabbitmq
from logging_code import setup_logging

logger = setup_logging("CAMERA_MANAGER")


class CameraManager:

    def process_camera(self, cam):
        connection = None
        cap = None
        FPS = 4

        try:
            logger.info(f"Starting camera: {cam['camera_name']}")

            connection = pika.BlockingConnection(
                pika.ConnectionParameters(
                    host=rabbitmq["host"],
                    port=rabbitmq["port"],
                    heartbeat=600
                )
            )
            channel = connection.channel()

            # ✅ Declare exchange
            channel.exchange_declare(
                exchange=rabbitmq["exchange"],
                exchange_type=rabbitmq["exchange_type"],
                durable=rabbitmq["durable"]
            )

            # ✅ Each camera gets its own queue
            cam_name = cam["camera_name"]
            queue_name = f"{cam_name}_queue"

            channel.queue_declare(queue=queue_name, durable=True)
            channel.queue_bind(
                exchange=rabbitmq["exchange"],
                queue=queue_name,
                routing_key=cam_name
            )

            # ✅ Open video
            cap = cv2.VideoCapture(cam["camera_feed"])
            if not cap.isOpened():
                logger.error(f"Cannot open video: {cam['camera_feed']}")
                return

            frame_count = 0

            while True:
                start = time.time()
                ret, frame = cap.read()

                if not ret:
                    logger.warning(f"{cam_name} video ended — restarting")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue

                frame_count += 1
                frame = cv2.resize(frame, (640, 640))

                # ✅ Encode frame
                _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
                frame_b64 = base64.b64encode(buffer).decode()

                payload = {
                    "camera_id":      cam["camera_id"],
                    "camera_name":    cam_name,
                    "camera_ip":      cam["camera_ip"],
                    "camera_outcome": cam["camera_outcome"],
                    "frame":          frame_b64
                }

                # ✅ Send to camera specific queue
                channel.basic_publish(
                    exchange=rabbitmq["exchange"],
                    routing_key=cam_name,
                    body=json.dumps(payload)
                )

                logger.info(f"Sent frame {frame_count} from {cam_name}")

                # FPS control
                elapsed = time.time() - start
                time.sleep(max(0, (1 / FPS) - elapsed))

        except Exception as e:
            _, msg, tb = sys.exc_info()
            logger.error(f"Error line {tb.tb_lineno}: {msg}")

        finally:
            if cap:
                cap.release()
            if connection and connection.is_open:
                connection.close()