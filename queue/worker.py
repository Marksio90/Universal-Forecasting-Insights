from rq import Worker, Queue, Connection
import os, redis
listen=['default']
conn = redis.from_url(os.getenv('REDIS_URL','redis://localhost:6379/0'))
if __name__=='__main__':
    with Connection(conn): Worker(list(map(Queue, listen))).work(with_scheduler=True)
