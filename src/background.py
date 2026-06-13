from __future__ import annotations

import logging
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable

logger = logging.getLogger(__name__)


class BackgroundWorker:
    def __init__(self, max_workers: int = 2) -> None:
        self.executor = ThreadPoolExecutor(max_workers=max_workers)

    def submit(self, func: Callable[..., Any], *args: Any, **kwargs: Any) -> Future[Any]:
        logger.info("Submitting background task: %s", getattr(func, "__name__", str(func)))
        future = self.executor.submit(func, *args, **kwargs)
        future.add_done_callback(self._log_result)
        return future

    def _log_result(self, future: Future[Any]) -> None:
        try:
            result = future.result()
            logger.info("Background task completed: %s", result)
        except Exception as exc:
            logger.exception("Background task failed: %s", exc)

    def shutdown(self, wait: bool = True) -> None:
        logger.info("Shutting down background worker")
        self.executor.shutdown(wait=wait)
