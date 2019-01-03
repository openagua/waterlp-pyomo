#!/usr/bin/env python
import json
import pika
import os
from main import run_model

connection = pika.BlockingConnection(pika.ConnectionParameters(host='localhost'))
channel = connection.channel()

queue_name = 'openagua-model_secret'

channel.queue_declare(queue=queue_name)


def callback(ch, method, properties, body):

    try:
        message = json.loads(body)
        env = message.get('env', {})
        args = message.get('args')
        ably_token = message.get('ably_token')

        for key, value in env.items():
            os.environ[key] = value
        print(" [x] Running model with %r" % args)
        run_model(args_list=args, ably_token=ably_token)
    except:
        pass # fail silently for now


channel.basic_consume(callback, queue=queue_name, no_ack=True)

print(' [*] Waiting for messages. To exit press CTRL+C')
channel.start_consuming()
