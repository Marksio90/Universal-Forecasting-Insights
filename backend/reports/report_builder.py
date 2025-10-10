from __future__ import annotations
import pandas as pd, plotly.express as px, plotly.io as pio, pathlib
from weasyprint import HTML

def build_html_summary(df: pd.DataFrame, metrics: dict)->str:
    fig = px.histogram(df.select_dtypes(include='number').iloc[:,0]) if df.select_dtypes(include='number').shape[1]>0 else None
    png_path = None
    if fig:
        png_path = pathlib.Path('reports')/'hist.png'; png_path.parent.mkdir(exist_ok=True)
        pio.write_image(fig, str(png_path))
    html = f"""<html><body>
    <h1>DataGenius Raport</h1>
    <p>Shape: {df.shape}</p>
    <p>Metrics: {metrics}</p>
    {'<img src="'+str(png_path)+'" />' if png_path else ''}
    </body></html>"""
    out_html = pathlib.Path('reports')/'summary.html'; out_html.write_text(html, encoding='utf-8'); return str(out_html)

def build_pdf_from_html(html_path: str)->str:
    out_pdf = str(pathlib.Path('reports')/'summary.pdf')
    HTML(filename=html_path).write_pdf(out_pdf)
    return out_pdf
