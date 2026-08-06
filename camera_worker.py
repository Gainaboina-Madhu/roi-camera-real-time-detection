from threading import Thread
import sys
from camera_manager import CameraManager
from logging_code import setup_logging

logger = setup_logging("CAMERA_WORKER")

class CameraWorker:

    def __init__(self, cams):
        self.cams = cams

    def start(self):
        try:
            manager = CameraManager()
            threads = []

            for cam in self.cams:
                logger.info(f"Starting thread for {cam['camera_name']}")
                t = Thread(target=manager.process_camera, args=(cam,))
                t.start()
                threads.append(t)

            for t in threads:
                t.join()

        except Exception as e:
            _, msg, tb = sys.exc_info()
            logger.error(f"Error line {tb.tb_lineno}: {msg}")