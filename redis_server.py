import redis
import json
import sys
import cv2
import numpy as np
from config import redis_server
from logging_code import setup_logging

logger = setup_logging("REDIS_SERVER")


class RedisServer:

    def __init__(self):
        # For camera metadata (decode_responses=True for JSON strings)
        self.r = redis.Redis(
            host=redis_server["host"],
            port=redis_server["port"],
            db=redis_server["db"],
            decode_responses=True,
            socket_connect_timeout=5,
            protocol=2
        )
        # For binary frame data (decode_responses=False for raw bytes)
        self.r_binary = redis.Redis(
            host=redis_server["host"],
            port=redis_server["port"],
            db=redis_server["db"],
            decode_responses=False,
            socket_connect_timeout=5,
            protocol=2
        )
        self.json_file = redis_server["json_file"]

    def load_json_to_redis(self):
        try:
            self.r.ping()
            logger.info("Redis Connected Successfully")

            with open(self.json_file, "r") as f:
                data = json.load(f)

            for _, cam in data.items():
                self.r.set(cam["camera_id"], json.dumps(cam))
                logger.info(f"Loaded {cam['camera_name']} into Redis")

            return data

        except Exception as e:
            _, msg, tb = sys.exc_info()
            logger.error(f"Error line {tb.tb_lineno}: {msg}")
            return {}

    def save_frame_to_redis(self, camera_id, frame, quality=85):
        """
        Save a detection frame (numpy array) to Redis as JPEG bytes.
        Key format: frame:<camera_id>
        """
        try:
            # Encode frame as JPEG
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
            _, buffer = cv2.imencode(".jpg", frame, encode_params)
            frame_bytes = buffer.tobytes()

            # Save to Redis — overwrites previous frame
            key = f"frame:{camera_id}"
            self.r_binary.set(key, frame_bytes)
            logger.info(f"Frame saved to Redis for camera: {camera_id}")

        except Exception as e:
            _, msg, tb = sys.exc_info()
            logger.error(f"Error line {tb.tb_lineno}: {msg}")

    def get_frame_from_redis(self, camera_id):
        """
        Retrieve the latest frame for a camera from Redis.
        Returns a numpy array (BGR) or None if not found.
        """
        try:
            key = f"frame:{camera_id}"
            frame_bytes = self.r_binary.get(key)

            if frame_bytes is None:
                logger.warning(f"No frame found in Redis for camera: {camera_id}")
                return None

            # Decode JPEG bytes back to numpy array
            np_array = np.frombuffer(frame_bytes, dtype=np.uint8)
            frame = cv2.imdecode(np_array, cv2.IMREAD_COLOR)
            return frame

        except Exception as e:
            _, msg, tb = sys.exc_info()
            logger.error(f"Error line {tb.tb_lineno}: {msg}")
            return None

    def get_camera(self, camera_id):
        try:
            data = self.r.get(camera_id)
            return json.loads(data) if data else None
        except Exception as e:
            _, msg, tb = sys.exc_info()
            logger.error(f"Error line {tb.tb_lineno}: {msg}")
            return None