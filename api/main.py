# api/main.py
# === IMPORTY I KONFIG ===
from __future__ import annotations
from typing import Optional, Dict, Any, List
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends, Request
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from jose import jwt, JWTError
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from passlib.context import CryptContext
from pydantic import BaseModel
from functools import lru_cache
import pandas as pd
import joblib, io, os, asyncio, json, time, numpy as np

# (opcjonalnie) CORS
# from fastapi.middleware.cors import CORSMiddleware

try:
    from loguru import logger
except Exception:
    # fallback minimal logger
    import logging as logger  # type: ignore

from backend.monitoring.drift import population_stability_index
from backend.monitoring.alerts import send_slack, send_email

# === USTAWIENIA / SECRETS ===
JWT_SECRET = os.getenv("JWT_SECRET", "change_me_super_secret")
JWT_EXPIRE_MIN = int(os.getenv("JWT_EXPIRE_MIN", "60"))
ALGORITHM = "HS256"

PSI_WARN = float(os.getenv("PSI_THRESHOLD_WARN", "0.10"))
PSI_ALERT = float(os.getenv("PSI_THRESHOLD_ALERT", "0.25"))

# === AUTORYZACJA / RBAC ===
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

# Uwaga: wartości demo; w produkcji ładuj z DB/Secrets!
USERS = {
    "admin":   {"password": pwd_ctx.hash("admin123"),   "role": "admin"},
    "analyst": {"password": pwd_ctx.hash("analyst123"), "role": "analyst"},
    "viewer":  {"password": pwd_ctx.hash("viewer123"),  "role": "viewer"},
}

def create_access_token(data: dict, minutes: int = JWT_EXPIRE_MIN) -> str:
    to_encode = data.copy()
    to_encode.update({"exp": datetime.utcnow() + timedelta(minutes=minutes)})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=ALGORITHM)

def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, str]:
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        username: Optional[str] = payload.get("sub")
        role: Optional[str] = payload.get("role")
        if not username or not role:
            raise HTTPException(status_code=401, detail="Invalid token payload")
        return {"username": username, "role": role}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

def require_role(*roles: str):
    def wrapper(user=Depends(get_current_user)):
        if user["role"] not in roles:
            raise HTTPException(status_code=403, detail="Forbidden")
        return user
    return wrapper

# === MODELE ODPOWIEDZI ===
class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    role: str

class PredictResponse(BaseModel):
    n: int
    predictions: List[Any]

class ErrorResponse(BaseModel):
    detail: str

# === APLIKACJA ===
app = FastAPI(title="DataGenius API v9.1 PRO", version="9.1.0")

# (opcjonalnie) CORS:
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # ustaw na listę dozwolonych domen w PROD
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# === GLOBALNY SCHEDULER (START/STOP) ===
sched: Optional[BackgroundScheduler] = None
base_ref: Optional[pd.Series] = None

@app.on_event("startup")
def _on_startup():
    global sched
    if isinstance(logger, type):
        # fallback logging
        pass
    else:
        logger.add("logs/api.log", rotation="5 MB", retention="7 days", enqueue=True, level="INFO")
        logger.info("API startup: initializing scheduler")
    if sched is None:
        s = BackgroundScheduler()
        # rejestracja jobu driftu
        s.add_job(drift_job, "interval", minutes=60, id="drift-monitor", replace_existing=True)
        s.start()
        globals()["sched"] = s

@app.on_event("shutdown")
def _on_shutdown():
    global sched
    if sched:
        if not isinstance(logger, type):
            logger.info("API shutdown: stopping scheduler")
        sched.shutdown(wait=False)
        sched = None

# === HANDLERY BŁĘDÓW ===
@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(status_code=exc.status_code, content=ErrorResponse(detail=str(exc.detail)).model_json())

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(status_code=422, content=ErrorResponse(detail="Validation error").model_json())

# === NARZĘDZIA / UTILS ===
@lru_cache(maxsize=16)
def _load_model_cached(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model file not found: {path}")
    t0 = time.time()
    model = joblib.load(path)
    if not isinstance(logger, type):
        logger.info(f"Loaded model {path} in {time.time()-t0:.3f}s (cached)")
    return model

def _read_dataframe(upload: UploadFile) -> pd.DataFrame:
    name = (upload.filename or "").lower()
    content = upload.file.read()
    if name.endswith(".parquet"):
        return pd.read_parquet(io.BytesIO(content))
    # fallback CSV
    return pd.read_csv(io.BytesIO(content))

def _validate_features(model, df: pd.DataFrame):
    # miękka walidacja: jeśli model ma n_features_in_ i/lub cechy z preprocesora
    try:
        if hasattr(model, "n_features_in_") and df.shape[1] != int(model.n_features_in_):
            raise HTTPException(status_code=400, detail=f"Feature count mismatch: df={df.shape[1]} vs model={int(model.n_features_in_)}")
    except Exception:
        # brak atrybutu — ignorujemy
        pass

# === ENDPOINTY ===
@app.get("/health")
def health():
    return {"status": "ok", "version": app.version, "time_utc": datetime.utcnow().isoformat()}

@app.post("/auth/token", response_model=TokenResponse, responses={401: {"model": ErrorResponse}})
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = USERS.get(form_data.username)
    if not user or not pwd_ctx.verify(form_data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Incorrect credentials")
    token = create_access_token({"sub": form_data.username, "role": user["role"]})
    return TokenResponse(access_token=token, role=user["role"])

@app.post(
    "/predict",
    response_model=PredictResponse,
    responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}, 403: {"model": ErrorResponse}},
)
async def predict(
    model_path: str = Form(...),
    file: UploadFile = File(...),
    user=Depends(require_role("admin", "analyst")),
):
    t0 = time.time()
    try:
        model = _load_model_cached(model_path)
        df = _read_dataframe(file)
        _validate_features(model, df)
        preds = model.predict(df)
        if not isinstance(logger, type):
            logger.info(f"/predict user={user['username']} n={len(preds)} dt={time.time()-t0:.3f}s")
        return PredictResponse(n=int(len(preds)), predictions=preds.tolist())
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Predict error: {e}")

@app.post(
    "/predict/stream",
    responses={400: {"model": ErrorResponse}, 401: {"model": ErrorResponse}, 403: {"model": ErrorResponse}},
)
async def predict_stream(
    model_path: str = Form(...),
    file: UploadFile = File(...),
    chunks: int = Form(10),
    user=Depends(require_role("admin", "analyst")),
):
    """Strumień NDJSON: każda linia to JSON {chunk, n, sample_pred}."""
    try:
        model = _load_model_cached(model_path)
        df = _read_dataframe(file)
        _validate_features(model, df)
        parts = np.array_split(df, max(1, int(chunks)))

        async def gen():
            for i, chunk in enumerate(parts):
                y = model.predict(chunk)
                payload = {"chunk": i, "n": int(len(y)), "sample_pred": y[:5].tolist()}
                yield json.dumps(payload) + "\n"
                await asyncio.sleep(0)  # pozwól event loopowi oddychać

        return StreamingResponse(gen(), media_type="application/x-ndjson")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Predict stream error: {e}")

# === DRIFT MONITORING (PSI) ===
def drift_job():
    """Prosty przykład: porównujemy pierwszą kolumnę numeryczną z 'reports/tmp_train.csv' do 'reports/tmp_curr.csv'."""
    global base_ref
    try:
        train_path = "reports/tmp_train.csv"
        curr_path = "reports/tmp_curr.csv"
        if base_ref is None and os.path.exists(train_path):
            dfb = pd.read_csv(train_path).select_dtypes(include="number")
            if dfb.shape[1] == 0:
                return
            base_ref = dfb.iloc[:, 0]
        if base_ref is None:
            return
        if os.path.exists(curr_path):
            dfc = pd.read_csv(curr_path).select_dtypes(include="number")
            if dfc.shape[1] == 0:
                return
            psi = population_stability_index(base_ref, dfc.iloc[:, 0])
            msg = f"PSI={psi:.3f} (warn={PSI_WARN}, alert={PSI_ALERT})"
            if not isinstance(logger, type):
                logger.info(f"[drift] {msg}")
            if psi >= PSI_ALERT:
                send_slack(f"🚨 PSI ALERT {psi:.3f}")
                send_email("PSI Alert", f"PSI exceeded: {psi:.3f}", "ops@example.com")
            elif psi >= PSI_WARN:
                send_slack(f"⚠️ PSI WARNING {psi:.3f}")
    except Exception as e:
        if not isinstance(logger, type):
            logger.warning(f"[drift] error: {e}")
        # nie przerywamy schedulera
        return
