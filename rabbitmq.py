import pika

# RabbitMQ Configuration
RABBITMQ_HOST = "localhost"      # Change if your RabbitMQ server is on another host
QUEUE_NAME = "task_queue"        # Change to your queue name

def get_connection():
    connection = pika.BlockingConnection(
        pika.ConnectionParameters(host=RABBITMQ_HOST)
    )

    channel = connection.channel()

    channel.queue_declare(queue=QUEUE_NAME, durable=True)

    return connection, channel


if __name__ == "__main__":
    connection, channel = get_connection()
    print("Connected to RabbitMQ successfully!")
    connection.close()