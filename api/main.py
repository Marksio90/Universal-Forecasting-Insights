from __future__ import annotations
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from fastapi.responses import StreamingResponse
from jose import jwt, JWTError
from datetime import datetime, timedelta
from apscheduler.schedulers.background import BackgroundScheduler
from passlib.context import CryptContext
import pandas as pd, joblib, io, os, time, asyncio
from backend.monitoring.drift import population_stability_index
from backend.monitoring.alerts import send_slack, send_email

JWT_SECRET = os.getenv("JWT_SECRET","change_me_super_secret")
JWT_EXPIRE_MIN = int(os.getenv("JWT_EXPIRE_MIN","60"))
ALGORITHM = "HS256"
pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")
USERS = {
    "admin": {"password": pwd_ctx.hash("admin123"), "role": "admin"},
    "analyst": {"password": pwd_ctx.hash("analyst123"), "role": "analyst"},
    "viewer": {"password": pwd_ctx.hash("viewer123"), "role": "viewer"}
}
def create_access_token(data: dict, minutes: int = JWT_EXPIRE_MIN):
    to_encode = data.copy(); to_encode.update({"exp": datetime.utcnow()+timedelta(minutes=minutes)})
    return jwt.encode(to_encode, JWT_SECRET, algorithm=ALGORITHM)
def get_current_user(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        username: str = payload.get("sub"); role: str = payload.get("role")
        if not username or not role: raise HTTPException(status_code=401, detail="Invalid token")
        return {"username": username, "role": role}
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")
def require_role(*roles):
    def wrapper(user=Depends(get_current_user)):
        if user["role"] not in roles: raise HTTPException(status_code=403, detail="Forbidden")
        return user
    return wrapper

app = FastAPI(title="DataGenius API v9.1", version="9.1")
sched = BackgroundScheduler(); sched.start()

@app.get("/health")
def health(): return {"status":"ok"}

@app.post("/auth/token")
def login(form_data: OAuth2PasswordRequestForm = Depends()):
    user = USERS.get(form_data.username)
    if not user or not pwd_ctx.verify(form_data.password, user["password"]):
        raise HTTPException(status_code=401, detail="Incorrect credentials")
    token = create_access_token({"sub": form_data.username, "role": user["role"]})
    return {"access_token": token, "token_type": "bearer", "role": user["role"]}

@app.post("/predict")
async def predict(model_path: str = Form(...), file: UploadFile = File(...), user=Depends(require_role("admin","analyst"))):
    try:
        model = joblib.load(model_path)
        df = pd.read_csv(io.BytesIO(await file.read()))
        preds = model.predict(df)
        return {"n": int(len(preds)), "predictions": preds.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Predict error: {e}")

@app.post("/predict/stream")
async def predict_stream(model_path: str = Form(...), file: UploadFile = File(...), user=Depends(require_role("admin","analyst"))):
    try:
        model = joblib.load(model_path)
        df = pd.read_csv(io.BytesIO(await file.read()))
        async def gen():
            for i,chunk in enumerate(np.array_split(df, 10)):
                y = model.predict(chunk)
                yield (str({"chunk": i, "n": int(len(y)), "pred": y[:5].tolist()}) + "\n")
                await asyncio.sleep(0.05)
        return StreamingResponse(gen(), media_type="text/plain")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Predict stream error: {e}")

# Drift scheduler example (hourly)
base_ref = None
def drift_job():
    global base_ref
    try:
        import pandas as pd
        if base_ref is None and os.path.exists("reports/tmp_train.csv"):
            base_ref = pd.read_csv("reports/tmp_train.csv").select_dtypes(include='number').iloc[:,0]
        if base_ref is None: return
        if os.path.exists("reports/tmp_curr.csv"):
            curr = pd.read_csv("reports/tmp_curr.csv").select_dtypes(include='number').iloc[:,0]
            psi = population_stability_index(base_ref, curr)
            if psi>0.25:
                send_slack(f"🚨 PSI ALERT {psi:.3f}")
                send_email("PSI Alert", f"PSI exceeded: {psi:.3f}", "ops@example.com")
    except Exception:
        pass

sched.add_job(drift_job, "interval", minutes=60, id="drift-monitor", replace_existing=True)
