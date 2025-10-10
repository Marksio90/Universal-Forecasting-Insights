from __future__ import annotations
import os, smtplib, ssl, requests
from email.mime.text import MIMEText

def send_slack(text: str)->bool:
    url=os.getenv("SLACK_WEBHOOK_URL")
    if not url: return False
    try:
        r=requests.post(url, json={"text": text}, timeout=5)
        return r.status_code==200
    except Exception: return False

def send_email(subject: str, body: str, to: str)->bool:
    host=os.getenv("SMTP_HOST"); port=int(os.getenv("SMTP_PORT","587"))
    user=os.getenv("SMTP_USER"); pwd=os.getenv("SMTP_PASSWORD")
    if not host or not user or not pwd: return False
    msg=MIMEText(body, "plain", "utf-8"); msg["Subject"]=subject; msg["From"]=user; msg["To"]=to
    try:
        ctx=ssl.create_default_context()
        with smtplib.SMTP(host, port) as s:
            s.starttls(context=ctx); s.login(user,pwd); s.sendmail(user,[to], msg.as_string())
        return True
    except Exception: return False
