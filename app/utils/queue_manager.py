from queue import Queue

task_queue = Queue()

def queue_size() -> int:
    """현재 대기 중인 이미지 작업 개수 반환"""
    return task_queue.qsize()