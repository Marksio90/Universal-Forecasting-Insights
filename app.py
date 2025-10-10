from __future__ import annotations
import os, streamlit as st, pandas as pd, redis
from rq import Queue
from dotenv import load_dotenv
from src.utils.logger import configure_logger
from src.utils.helpers import smart_read
from src.data_processing.data_cleaner import clean_dataframe
from visualization.charts import corr_heatmap
from frontend.ui import kpi_row
from backend.automl_fusion import train_fusion
from backend.model_io import save_model
from backend.reports.report_builder import build_html_summary, build_pdf_from_html
from backend.feature_store.store import FeatureStore
from chat.agent import chat_reply
from src.utils.jwt_utils import decode_role

load_dotenv(); logger=configure_logger()
st.set_page_config(page_title="DataGenius — v9.1 ULTRA COMPLETE+", page_icon="🧠", layout="wide")
st.title("DataGenius v9.1 — ULTRA COMPLETE+")

if "df" not in st.session_state: st.session_state.df=None
if "history" not in st.session_state: st.session_state.history=[]

with st.sidebar:
    st.subheader("🔐 Access & Secrets")
    k = st.text_input("OpenAI API Key", type="password")
    if k: os.environ["OPENAI_API_KEY"]=k
    jwt = st.text_input("JWT token (RBAC)", type="password")
    role = decode_role(jwt, os.getenv("JWT_SECRET","change_me_super_secret")) if jwt else None
    st.caption(f"Rola: {role or 'viewer'} (admin/analyst/viewer)")
    st.caption("Sidebar służy tylko do sekretów i uprawnień.")

tabs_all = ["📤 Upload & Prep","🔍 EDA","🚀 AutoML FUSION","🗂️ Feature Store","🧶 Jobs (RQ)","📄 Reports","💬 Chat"]
def has_access(tab: str)->bool:
    if role=="admin": return True
    if role=="analyst": return tab in ["📤 Upload & Prep","🔍 EDA","🚀 AutoML FUSION","🗂️ Feature Store","🧶 Jobs (RQ)","📄 Reports","💬 Chat"]
    return tab in ["📤 Upload & Prep","🔍 EDA","🗂️ Feature Store","📄 Reports","💬 Chat"]
tabs = [t for t in tabs_all if has_access(t)]
tab_objs = st.tabs(tabs)
tab = {n:o for n,o in zip(tabs, tab_objs)}

with tab["📤 Upload & Prep"]:
    up=st.file_uploader("Wrzuć CSV/Parquet/Excel", type=["csv","parquet","xlsx","xls"])
    if up:
        try:
            df=smart_read(up); st.session_state.df = clean_dataframe(df); st.success(f"Wczytano: {df.shape}")
            st.dataframe(st.session_state.df.head(50))
        except Exception as e: st.error(f"Błąd: {e}")

with tab["🔍 EDA"]:
    if st.session_state.df is None: st.info("Najpierw upload.")
    else:
        df=st.session_state.df
        st.write("Dtypes:", dict(df.dtypes.astype(str)))
        fig=corr_heatmap(df)
        if fig: st.plotly_chart(fig, use_container_width=True)

if "🚀 AutoML FUSION" in tab:
    with tab["🚀 AutoML FUSION"]:
        if st.session_state.df is None: st.info("Najpierw upload.")
        else:
            df=st.session_state.df
            target=st.selectbox("Target", options=list(df.columns), index=(len(df.columns)-1))
            trials=st.slider("Próby (Optuna)", 15, 80, 35, step=5)
            if st.button("Start FUSION (lokalnie)"):
                with st.spinner("Trening..."):
                    res=train_fusion(df, target=target, trials=trials)
                kpi_row({"Problem": res.problem_type, res.metric_name.upper(): f"{res.best_score:.4f}"})
                st.write("🏆 Leaderboard:", res.leaderboard)
                if res.blend_weights: st.write("⚖️ Blend weights:", res.blend_weights)
                path=save_model(res.model, f"FUSION_{res.problem_type}")
                st.success(f"Zapisano model: {path}")
                html = build_html_summary(df, {res.metric_name: res.best_score}); pdf = build_pdf_from_html(html)
                st.success(f"Raporty zapisane: {html}, {pdf}")

with tab["🗂️ Feature Store"]:
    st.subheader("Light Feature Store (Parquet)")
    fs = FeatureStore()
    if st.session_state.df is None: st.info("Najpierw upload.")
    else:
        name = st.text_input("Nazwa feature setu", value="dataset")
        if st.button("Zapisz jako wersję"):
            path = fs.write(st.session_state.df, name); st.success(f"Zapisano: {path}")
        if st.button("Wczytaj najnowszą"):
            try:
                df = fs.read_latest(name); st.dataframe(df.head(50))
            except Exception as e:
                st.error(f"Brak wersji: {e}")

if "🧶 Jobs (RQ)" in tab:
    with tab["🧶 Jobs (RQ)"]:
        if st.session_state.df is None: st.info("Najpierw upload.")
        else:
            df=st.session_state.df
            redis_url=os.getenv("REDIS_URL","redis://localhost:6379/0")
            try:
                import redis as _r
                r=_r.from_url(redis_url); q=Queue('default', connection=r)
                tmp_csv="reports/tmp_train.csv"; os.makedirs("reports", exist_ok=True); df.to_csv(tmp_csv, index=False)
                target=st.selectbox("Target (jobs)", options=list(df.columns), index=(len(df.columns)-1), key="job_target")
                trials=st.slider("Próby (jobs)", 10, 80, 30, step=5, key="job_trials")
                if st.button("Wyślij job treningu"):
                    job=q.enqueue("queue.jobs.train_job", tmp_csv, target, trials, job_timeout=60*60)
                    st.success(f"Wysłano job: {job.id}")
            except Exception as e:
                st.warning(f"Kolejka niedostępna: {e}")

with tab["📄 Reports"]:
    st.write("Raporty generowane są automatycznie po treningu (HTML + PDF w folderze `reports/`).")

with tab["💬 Chat"]:
    mode = st.radio("Tryb eksperta", ["Data Assistant","Modeling Coach","Forecasting Guru"], horizontal=True)
    for role_msg,content in st.session_state.history:
        with st.chat_message(role_msg): st.markdown(content)
    prompt = st.chat_input("Zadaj pytanie… (np. 'pokaż schema')")
    if prompt:
        st.session_state.history.append(("user", prompt))
        with st.chat_message("user"): st.markdown(prompt)
        reply = chat_reply([{"role":r,"content":c} for r,c in st.session_state.history], mode, st.session_state.df)
        st.session_state.history.append(("assistant", reply))
        with st.chat_message("assistant"): st.markdown(reply)
