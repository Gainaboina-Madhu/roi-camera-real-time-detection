import multiprocessing
import sys
from redis_server import RedisServer
from processor_manager import ProcessorManager
from logging_code import setup_logging

logger = setup_logging("MAIN")

if __name__ == "__main__":
    multiprocessing.set_start_method("spawn")
    try:
        logger.info("Loading cameras into Redis...")
        redis_obj = RedisServer()
        cameras = redis_obj.load_json_to_redis()

        if not cameras:  # ← Fix: guard against empty/None result
            logger.error("No cameras loaded. Check Redis connection and JSON file.")
            sys.exit(1)

        logger.info(f"Loaded {len(cameras)} cameras")

        processor = ProcessorManager(cameras)
        processor.start()

    except Exception as e:
        _, msg, tb = sys.exc_info()
        logger.error(f"Error line {tb.tb_lineno}: {msg}")