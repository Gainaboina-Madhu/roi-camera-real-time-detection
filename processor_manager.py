from multiprocessing import Process
from camera_worker import CameraWorker
from logging_code import setup_logging
import sys

logger = setup_logging("PROCESSOR_MANAGER")

class ProcessorManager:

    def __init__(self, cameras):
        self.cameras = list(cameras.values())

    def run_process(self, cam_subset):
        worker = CameraWorker(cam_subset)
        worker.start()

    def start(self):
        try:
            processes = []

            # 2 cameras per process → 2 processes for 4 cameras
            for i in range(0, len(self.cameras), 2):
                subset = self.cameras[i:i + 2]
                logger.info(f"Launching process for cameras: {[c['camera_name'] for c in subset]}")
                p = Process(target=self.run_process, args=(subset,))
                p.start()
                processes.append(p)

            for p in processes:
                p.join()

        except Exception as e:
            _, msg, tb = sys.exc_info()
            logger.error(f"Error line {tb.tb_lineno}: {msg}")