import pathlib, joblib, time
pathlib.Path("models").mkdir(exist_ok=True)
def save_model(model, name:str)->str:
    p=pathlib.Path("models")/f"{int(time.time())}_{name}.joblib"
    joblib.dump(model, p); return str(p)
