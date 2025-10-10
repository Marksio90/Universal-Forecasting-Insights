# queue/worker.py
# === WORKER_RQ_PRO+++ ===
from __future__ import annotations
import os, sys, time, json, signal, socket, threading, logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from dataclasses import dataclass
from typing import List

import redis
from rq import Worker, Queue, Connection

# --- optional integrations (safe import) ---
try:
    import sentry_sdk  # type: ignore
except Exception:
    sentry_sdk = None  # type: ignore

try:
    from prometheus_client import Counter, Histogram, start_http_server  # type: ignore
except Exception:
    Counter = Histogram = None  # type: ignore
    def start_http_server(*_, **__):  # type: ignore
        return

# ============ CONFIG ============
@dataclass
class Cfg:
    redis_url: str
    queues: List[str]
    worker_name: str
    with_scheduler: bool
    burst: bool
    log_level: str
    json_logs: bool
    health_port: int
    prom_port: int
    default_job_timeout: int

def _env(key: str, default: str) -> str:
    v = os.getenv(key, default)
    return v.strip() if isinstance(v, str) else default

def _build_cfg() -> Cfg:
    # Allow REDIS_URL or REDIS_HOST/PORT/DB
    redis_url = _env("REDIS_URL", "")
    if not redis_url:
        host = _env("REDIS_HOST", "localhost")
        port = _env("REDIS_PORT", "6379")
        db   = _env("REDIS_DB", "0")
        redis_url = f"redis://{host}:{port}/{db}"

    queues = [q.strip() for q in _env("RQ_QUEUES", "high,default,low").split(",") if q.strip()]
    worker_name = _env("RQ_WORKER_NAME", f"{socket.gethostname()}:{os.getpid()}")
    with_scheduler = _env("RQ_WITH_SCHEDULER", "1") in {"1","true","TRUE","yes"}
    burst = _env("RQ_BURST", "0") in {"1","true","TRUE","yes"}
    log_level = _env("LOG_LEVEL", "INFO").upper()
    json_logs = _env("LOG_JSON", "0") in {"1","true","TRUE","yes"}
    health_port = int(_env("HEALTH_PORT", "8088"))
    prom_port   = int(_env("PROM_PORT", "9108"))
    default_job_timeout = int(_env("RQ_JOB_TIMEOUT", "7200"))  # 2h default (trenowanie ML bywa długie)
    return Cfg(redis_url, queues, worker_name, with_scheduler, burst, log_level, json_logs, health_port, prom_port, default_job_timeout)

CFG = _build_cfg()

# ============ LOGGING ============
class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:  # type: ignore[override]
        base = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            "lvl": record.levelname,
            "msg": record.getMessage(),
            "logger": record.name,
        }
        if record.exc_info:
            base["exc"] = self.formatException(record.exc_info)
        return json.dumps(base, ensure_ascii=False)

def _setup_logging():
    h = logging.StreamHandler(sys.stdout)
    if CFG.json_logs:
        h.setFormatter(_JsonFormatter())
    else:
        h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(h)
    root.setLevel(CFG.log_level)

_setup_logging()
log = logging.getLogger("worker")

# ============ SENTRY (opcjonalnie) ============
def _setup_sentry():
    dsn = _env("SENTRY_DSN", "")
    if dsn and sentry_sdk is not None:
        sentry_sdk.init(dsn=dsn, traces_sample_rate=float(_env("SENTRY_TRACES", "0.0")))
        log.info("Sentry enabled")
_setup_sentry()

# ============ PROMETHEUS (opcjonalnie) ============
if Counter and Histogram:
    MET_START   = Counter("rq_jobs_started_total",   "Jobs started",   ["queue"])
    MET_OK      = Counter("rq_jobs_succeeded_total", "Jobs succeeded", ["queue"])
    MET_FAIL    = Counter("rq_jobs_failed_total",    "Jobs failed",    ["queue"])
    MET_DUR     = Histogram("rq_job_duration_seconds","Job duration seconds", ["queue","func"])
    try:
        start_http_server(CFG.prom_port)
        log.info("Prometheus metrics on :%d", CFG.prom_port)
    except Exception as e:
        log.warning("Prometheus start failed: %s", e)
else:
    MET_START = MET_OK = MET_FAIL = MET_DUR = None  # type: ignore

# ============ HEALTHZ ============
class _HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):  # type: ignore[override]
        if self.path == "/healthz":
            self.send_response(200); self.end_headers(); self.wfile.write(b"ok")
        else:
            self.send_response(404); self.end_headers()
    def log_message(self, *args, **kwargs):  # silence
        return

def _start_health_server():
    try:
        srv = HTTPServer(("0.0.0.0", CFG.health_port), _HealthHandler)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        log.info("Healthz on :%d", CFG.health_port)
    except Exception as e:
        log.warning("Health server failed: %s", e)

_start_health_server()

# ============ REDIS / QUEUES ============
def _redis_conn():
    # Keepalives help in containers
    return redis.from_url(CFG.redis_url, socket_keepalive=True, health_check_interval=30)

def _queues(conn) -> List[Queue]:
    # FIFO per queue; order defines priority
    return [Queue(name=q, connection=conn, default_timeout=CFG.default_job_timeout) for q in CFG.queues]

# ============ WORKER (metrics + graceful) ============
class MetricsWorker(Worker):
    """
    Worker rozszerzony o metryki Prometheus i czytelne logi cyklu życia joba.
    """
    def perform_job(self, job, queue, heartbeat_ttl=None):  # type: ignore[override]
        qname = getattr(queue, "name", "default")
        func  = getattr(job, "func_name", getattr(job, "description", "job"))
        start = time.time()
        if MET_START: MET_START.labels(qname).inc()
        log.info("Job start: id=%s queue=%s func=%s", job.id, qname, func)
        ok = super().perform_job(job, queue, heartbeat_ttl=heartbeat_ttl)
        dur = max(0.0, time.time() - start)
        if ok:
            if MET_OK: MET_OK.labels(qname).inc()
            if MET_DUR: MET_DUR.labels(qname, str(func)).observe(dur)
            log.info("Job done:  id=%s queue=%s func=%s dur=%.3fs", job.id, qname, func, dur)
        else:
            if MET_FAIL: MET_FAIL.labels(qname).inc()
            log.error("Job fail:  id=%s queue=%s func=%s dur=%.3fs", job.id, qname, func, dur)
        return ok

_STOP_REQUESTED = False
def _handle_signal(sig, frame):
    global _STOP_REQUESTED
    _STOP_REQUESTED = True
    log.warning("Signal %s received — graceful stop requested.", sig)

for s in (signal.SIGTERM, signal.SIGINT):
    signal.signal(s, _handle_signal)

# ============ MAIN ============
def main():
    log.info("Starting RQ worker | queues=%s | redis=%s | name=%s", CFG.queues, CFG.redis_url, CFG.worker_name)
    with Connection(_redis_conn()):
        queues = _queues(Connection())
        worker = MetricsWorker(queues, name=CFG.worker_name)
        # Make sure we stop gracefully after current job
        def _stopping():
            return _STOP_REQUESTED
        worker.should_stop = _stopping  # type: ignore[attr-defined]
        worker.work(with_scheduler=CFG.with_scheduler, burst=CFG.burst)

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.exception("Worker crashed: %s", e)
        sys.exit(1)
