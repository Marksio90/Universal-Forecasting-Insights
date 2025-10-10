from __future__ import annotations
import time
from typing import Callable

def rate_limited(min_interval: float):
    def deco(func: Callable):
        last = [0.0]
        def wrapper(*a, **kw):
            now = time.time()
            delta = now - last[0]
            if delta < min_interval:
                time.sleep(min_interval - delta)
            last[0] = time.time()
            return func(*a, **kw)
        return wrapper
    return deco
