from __future__ import annotations
from typing import Dict, Any
from weasyprint import HTML

BASE_TMPL = """
<!doctype html>
<html lang="pl">
<head>
<meta charset="utf-8">
<title>{title}</title>
<style>
body {{ font-family: -apple-system, Segoe UI, Roboto, Arial; margin: 24px; }}
h1 {{ color:#333 }}
.kpi-grid {{ display:grid; grid-template-columns: repeat(3,1fr); gap:12px; }}
.kpi {{ border:1px solid #ddd; border-radius:10px; padding:12px; }}
.section {{ margin:16px 0 }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="section">
  <strong>Autor:</strong> {author} |
  <strong>Firma:</strong> {company}
</div>
<div class="section">
  <h2>KPI</h2>
  <div class="kpi-grid">
    {kpi_cards}
  </div>
</div>
<div class="section">
  <h2>Notatki</h2>
  <div>{notes_html}</div>
</div>
</body>
</html>
"""

def build_pdf(path_out: str, context: Dict[str, Any]) -> str:
    kpi_cards = "".join([f'<div class="kpi"><div><b>{k}</b></div><div>{v}</div></div>' for k,v in context.get("kpi",{}).items()])
    html = BASE_TMPL.format(
        title=context.get("title","Raport"),
        author=context.get("author",""),
        company=context.get("company",""),
        kpi_cards=kpi_cards,
        notes_html=context.get("notes","")
    )
    HTML(string=html).write_pdf(path_out)
    return path_out
