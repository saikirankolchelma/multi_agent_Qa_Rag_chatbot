import redis
import json
import logging
from config import config

logger = logging.getLogger(__name__)

class RedisBus:
    """A scalable message bus using Redis lists for queueing."""
    def __init__(self):
        try:
            self.redis_client = redis.Redis(
                host=config.REDIS_HOST,
                port=config.REDIS_PORT,
                db=0,
                decode_responses=True
            )
            self.redis_client.ping()
            logger.info(f"Successfully connected to Redis at {config.REDIS_HOST}:{config.REDIS_PORT}")
        except redis.exceptions.ConnectionError as e:
            logger.error(f"Could not connect to Redis. Please ensure Redis server is running. Error: {e}")
            raise

    def send(self, message: dict):
        """Sends a message to a specific agent's queue."""
        receiver_queue = f"queue:{message['receiver']}"
        try:
            self.redis_client.rpush(receiver_queue, json.dumps(message))
            logger.debug(f"Sent message to {receiver_queue}: {message.get('type')}")
        except Exception as e:
            logger.error(f"Failed to send message to {receiver_queue}: {e}")

    def receive(self, agent_name: str, block: bool = True, timeout: int = 5) -> dict | None:
        """Receives a message from an agent's queue."""
        agent_queue = f"queue:{agent_name}"
        try:
            if block:
                message = self.redis_client.blpop(agent_queue, timeout=timeout)
                if message:
                    return json.loads(message[1]) # blpop returns a tuple (queue_name, message)
            else:
                message = self.redis_client.lpop(agent_queue)
                if message:
                    return json.loads(message)
        except Exception as e:
            logger.error(f"Failed to receive message from {agent_queue}: {e}")
        return None

    def is_empty(self, agent_name: str) -> bool:
        """Checks if an agent's queue is empty."""
        agent_queue = f"queue:{agent_name}"
        return self.redis_client.llen(agent_queue) == 0