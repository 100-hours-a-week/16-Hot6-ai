import logging, torch
from utils.clear_cache import clear_cache
from utils.queue_manager import task_queue

logger = logging.getLogger(__name__)

def shutdown_event(app):
    logger.info("[Shutdown] 서버 종료 감지. GPU 및 리소스 정리 시작...")

    # Task Queue 비우기
    if hasattr(task_queue, "queue"):
        with task_queue.mutex:
            task_queue.queue.clear()
        logger.info("[Shutdown] Task Queue 비움 완료")

    # app.state에 등록된 torch 모델 제거
    deleted_attrs = []
    for attr_name in list(vars(app.state)):
        attr = getattr(app.state, attr_name)
        # torch 모델 또는 유사 객체만 제거
        if isinstance(attr, torch.nn.Module):
            delattr(app.state, attr_name)
            deleted_attrs.append(attr_name)
        elif hasattr(attr, 'to') and callable(getattr(attr, 'to')):
            # torch 모델이 아닌데도 GPU 메모리를 점유할 수 있는 커스텀 객체 대응
            delattr(app.state, attr_name)
            deleted_attrs.append(attr_name)

    if deleted_attrs:
        logger.info(f"[Shutdown] 다음 속성 제거됨: {', '.join(deleted_attrs)}")

    # GPU 캐시 및 가비지 컬렉션
    clear_cache()
    logger.info("[Shutdown] GPU 메모리 캐시 해제 및 GC 완료")
