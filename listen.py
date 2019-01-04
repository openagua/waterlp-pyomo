#!/usr/bin/env python
import json
import pika
import os
from main import run_model


def callback(ch, method, properties, body):
    try:
        message = json.loads(body)
        env = message.get('env', {})
        args = message.get('args')
        kwargs = message.get('kwargs')

        for key, value in env.items():
            os.environ[key] = value
        print(" [x] Running model with %r" % args)
        run_model(args_list=args, **kwargs)
    except:
        pass  # fail silently for now


def start_listening(queue_name):
    connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
    channel = connection.channel()
    channel.queue_declare(queue=queue_name)
    channel.basic_consume(callback, queue=queue_name, no_ack=True)
    print(' [*] Waiting for messages. To exit press CTRL+C')
    channel.start_consuming()


if __name__ == '__main__':
    try:
        model_secret = os.environ['MODEL_SECRET']
        queue_name = 'model-{}'.format(model_secret)
        start_listening(queue_name)
    except:
        raise
