# === frontend/ui.py ===
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Union, Any, Tuple
import math
import streamlit as st

# === STYL / JEDNORAZOWY CSS ===
_CSS = """
:root{
  --bg:#0b0f19; --card:#121826; --muted:#8b95a7; --accent:#6C5CE7;
  --radius:14px; --border:rgba(255,255,255,.10); --text:#fff;
}
.main .block-container{padding-top:1.0rem; max-width:1200px;}
.section{padding:.8rem 1rem; border-radius:var(--radius); border:1px solid var(--border); margin:.8rem 0; background:var(--card)}
.kpi-wrap{display:flex; gap:.6rem; align-items:flex-start; flex-wrap:wrap}
.kpi{flex:1 1 180px; min-width:160px; border:1px solid rgba(108,92,231,.25); border-radius:12px; padding:.7rem .9rem;
     background:linear-gradient(180deg, rgba(108,92,231,.10), rgba(108,92,231,.03))}
.kpi .label{color:var(--muted); font-size:.82rem}
.kpi .value{font-weight:800; font-size:1.25rem; line-height:1.2}
.kpi .delta{font-size:.82rem; margin-top:.15rem}
.kpi .delta.up{color:#16a34a} .kpi .delta.down{color:#ef4444}
"""

def apply_css(extra: Optional[str] = None) -> None:
    if not st.session_state.get("_ui_css_applied"):
        st.markdown(f"<style>{_CSS}{extra or ''}</style>", unsafe_allow_html=True)
        st.session_state["_ui_css_applied"] = True

# === KPI MODEL ===
@dataclass
class KPI:
    label: str
    value: Union[int, float, str]
    delta: Optional[Union[int, float, str]] = None   # np. +3.1% / -2
    help: Optional[str] = None
    fmt: Optional[str] = None                         # np. ".2f", ",.0f", ".2%"

Number = Union[int, float]

def _format(val: Any, fmt: Optional[str]) -> str:
    if fmt is None or isinstance(val, str):
        return str(val)
    try:
        # obsługa procentów
        if fmt.endswith("%"):
            perc = float(val)
            digits = fmt[:-1] if fmt[:-1] else ".2"
            return f"{perc:{digits}f}%"
        return f"{float(val):{fmt}}"
    except Exception:
        return str(val)

def _delta_class(delta: Optional[Union[int, float, str]]) -> str:
    try:
        if isinstance(delta, str) and delta.endswith("%"):
            v = float(delta.replace("%", ""))
        else:
            v = float(delta) if delta is not None else 0.0
        return "up" if v >= 0 else "down"
    except Exception:
        return "up"

# === SEKCJE ===
def section(title: str, subtitle: Optional[str] = None, *, icon: Optional[str] = None) -> None:
    apply_css()
    ico = f"{icon} " if icon else ""
    st.markdown(
        f"""<div class="section">
  <div style="font-weight:800; font-size:1.05rem">{ico}{title}</div>
  {f'<div style="color:#8b95a7; margin-top:.25rem">{subtitle}</div>' if subtitle else ''}
</div>""",
        unsafe_allow_html=True,
    )

# === KPI ROW (auto-siatka + formatowanie + tooltip) ===
def kpi_row(items: Union[Dict[str, Any], Iterable[KPI]], *, default_fmt: Optional[str] = None) -> None:
    """
    Renderuje KPI w elastycznej siatce (wrap). Akceptuje:
      - dict: {"AUC":0.964, "F1":0.912, ...}
      - list[KPI(...)]
    """
    apply_css()

    # normalizacja
    if isinstance(items, dict):
        data: List[KPI] = [KPI(k, v, None, None, default_fmt) for k, v in items.items()]
    else:
        data = [k if isinstance(k, KPI) else KPI(str(k), k) for k in items]  # type: ignore

    if not data:
        return

    # render HTML (ładniejsze niż gołe st.metric)
    blocks: List[str] = []
    for k in data:
        val = _format(k.value, k.fmt or default_fmt)
        delta_html = ""
        if k.delta is not None:
            delta_html = f'<div class="delta {_delta_class(k.delta)}">{_format(k.delta, None)}</div>'
        help_attr = f'title="{k.help}"' if k.help else ""
        blocks.append(
            f'<div class="kpi" {help_attr}><div class="label">{k.label}</div><div class="value">{val}</div>{delta_html}</div>'
        )

    st.markdown('<div class="kpi-wrap">' + "".join(blocks) + "</div>", unsafe_allow_html=True)

# === PRZYKŁAD (uruchom: streamlit run frontend/ui.py) ===
if __name__ == "__main__":
    apply_css()
    section("KPI – demo", "Auto-siatka, delta, tooltip", icon="📊")
    kpi_row([
        KPI("AUC", 0.964, "+0.6pp", "Area Under Curve", ".3f"),
        KPI("F1", 0.912, "+0.01", "F1 (weighted)", ".3f"),
        KPI("RMSE", 1.23, "-0.04", "Root Mean Squared Error", ".2f"),
        KPI("Samples", 125_430, None, "liczba obserwacji", ",.0f"),
    ])
