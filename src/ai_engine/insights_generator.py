# ai_insights.py — TURBO PRO
from __future__ import annotations

import json
import math
import logging
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Literal, Tuple

import pandas as pd
from .openai_integrator import chat_completion

# =========================
# Logger
# =========================
def get_logger(name: str = "ai_insights", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(level)
        h = logging.StreamHandler()
        h.setLevel(level)
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        h.setFormatter(fmt)
        logger.addHandler(h)
        logger.propagate = False
    return logger

LOGGER = get_logger()

# =========================
# Stałe i domyślne limity
# =========================
MAX_PROFILE_COLS = 50          # maks. liczba kolumn rozważanych w profilu
MAX_PROFILE_ROWS = 200_000     # maks. wiersze do profilu (sampling head)
TOP_MISSING = 5
TOP_NUNIQUE = 5
TOP_NUMERIC_VAR = 5

# =========================
# Prompty / schemat
# =========================
INSIGHTS_SCHEMA_KEYS = ("summary", "top_insights", "recommendations", "risks")

SYSTEM_PROMPT = (
    "Jesteś seniorem Data Scientist i Analitykiem Biznesowym.\n"
    "Twoim zadaniem jest tworzyć zwięzłe, precyzyjne wnioski biznesowe na podstawie metadanych o zbiorze danych "
    "i celu użytkownika. Nie wymyślaj wartości z danych — używaj wyłącznie dostarczonych metryk/cech.\n\n"
    "Zwracaj odpowiedź w 100% POPRAWNYM JSON (bez zbędnego tekstu) o dokładnym schemacie:\n"
    "{\n"
    '  "summary": string,\n'
    '  "top_insights": string[],\n'
    '  "recommendations": string[],\n'
    '  "risks": string[]\n'
    "}\n"
    "Każdy element listy to krótka, konkretna myśl (maks. 180 znaków)."
)

# =========================
# Dataclasses (typy wyników)
# =========================
@dataclass(frozen=True)
class InsightsOptions:
    mode: Literal["standard", "executive", "technical"] = "standard"
    sample_rows_for_profile: int = 50_000
    max_profile_cols: int = MAX_PROFILE_COLS
    domain_hint: Optional[str] = None  # np. 'retail', 'manufacturing', 'finance'

@dataclass(frozen=True)
class InsightsPayload:
    summary: str
    top_insights: List[str]
    recommendations: List[str]
    risks: List[str]

@dataclass(frozen=True)
class InsightsResult:
    ok: bool
    data: Optional[InsightsPayload]
    raw_text: Optional[str]
    message: str

@dataclass(frozen=True)
class InsightsError:
    ok: bool
    code: str
    message: str
    raw_text: Optional[str] = None

# =========================
# Utils
# =========================
def _truncate_str(s: str, max_len: int) -> str:
    return s if len(s) <= max_len else (s[: max_len - 1] + "…")

def _list_of_str(x: Any) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        out = [str(i) for i in x if isinstance(i, (str, int, float))]
        return out
    # pojedynczy string → lista
    if isinstance(x, (str, int, float)):
        return [str(x)]
    return []

def _coerce_schema(obj: Any) -> Optional[InsightsPayload]:
    """Lekkie dopasowanie do wymaganego schematu."""
    if not isinstance(obj, dict):
        return None
    if not all(k in obj for k in INSIGHTS_SCHEMA_KEYS):
        return None
    summary = str(obj.get("summary", "")).strip()
    top_insights = _list_of_str(obj.get("top_insights"))
    recommendations = _list_of_str(obj.get("recommendations"))
    risks = _list_of_str(obj.get("risks"))
    return InsightsPayload(summary=summary, top_insights=top_insights, recommendations=recommendations, risks=risks)

def _safe_json_loads(text: str) -> Optional[dict]:
    try:
        return json.loads(text)
    except Exception:
        return None

def _cap_list_items(items: List[str], max_len: int = 180, max_items: int = 8) -> List[str]:
    items = [i.strip() for i in items if i and isinstance(i, str)]
    items = [(_truncate_str(i, max_len)) for i in items]
    return items[:max_items]

def _sample_for_profile(df: pd.DataFrame, n_rows: int) -> pd.DataFrame:
    if len(df) <= n_rows:
        return df
    # deterministyczny head (szybki, reprodukowalny)
    return df.head(n_rows)

# =========================
# Profil kontekstu danych
# =========================
def _profile_context(df: pd.DataFrame, max_cols: int = MAX_PROFILE_COLS, max_rows: int = MAX_PROFILE_ROWS) -> str:
    """
    Buduje bezpieczny, zwięzły profil danych (bez ujawniania wartości wierszy).
    Zwraca tekst używany w promptach.
    """
    df2 = df.iloc[:, :max_cols]
    df2 = _sample_for_profile(df2, min(max_rows, MAX_PROFILE_ROWS))

    info_lines: List[str] = []
    info_lines.append(f"- Wiersze: {len(df2):,} (z {len(df):,})")
    info_lines.append(f"- Kolumny: {df2.shape[1]:,} (z {df.shape[1]:,})")

    size = df2.size or 1
    missing_pct = float((df2.isna().sum().sum()) / size) * 100.0
    dupes = int(df2.duplicated().sum())
    info_lines.append(f"- Braki: {missing_pct:.2f}%")
    info_lines.append(f"- Duplikaty (na próbie): {dupes:,}")

    # Typy kolumn
    type_counts = df2.dtypes.astype(str).value_counts().to_dict()
    info_lines.append("- Typy kolumn: " + ", ".join([f"{t}: {n}" for t, n in type_counts.items()]))

    # Najczęstsze braki
    miss_cols = df2.isna().mean().sort_values(ascending=False).head(TOP_MISSING)
    if miss_cols.max() > 0:
        info_lines.append("- Najwięcej braków: " + ", ".join([f"{c} ({p:.1%})" for c, p in miss_cols.items()]))

    # Unikalność
    nunique_cols = df2.nunique(dropna=True).sort_values(ascending=False).head(TOP_NUNIQUE)
    info_lines.append("- Top unikalnych wartości: " + ", ".join([f"{c}: {int(n)}" for c, n in nunique_cols.items()]))

    # Wariancja liczbowych
    num_cols = df2.select_dtypes("number")
    if not num_cols.empty:
        var_rank = num_cols.var(numeric_only=True).sort_values(ascending=False).head(TOP_NUMERIC_VAR)
        info_lines.append("- Największa wariancja (num): " + ", ".join([f"{c}: {v:.2e}" for c, v in var_rank.items()]))

    return "\n".join(info_lines)

# =========================
# Prompty użytkownika
# =========================
def _build_user_prompt(context_summary: str, goal: Optional[str], mode: str, domain_hint: Optional[str]) -> str:
    goal_txt = goal.strip() if goal else "Brak sprecyzowanego celu."
    tone = {
        "standard": "styl zbalansowany: klarowny i biznesowy",
        "executive": "styl executive: bardzo zwięzły, nacisk na wpływ/ROI i KPI",
        "technical": "styl techniczny: precyzyjny, metryki i metody",
    }[mode if mode in ("standard", "executive", "technical") else "standard"]

    domain_line = f"Domena: {domain_hint}\n" if domain_hint else ""

    return f"""Dane (skrótowy profil):
{context_summary}

Cel użytkownika:
{goal_txt}

{domain_line}Tryb pisania: {tone}

Zadanie:
1) Napisz krótkie executive summary.
2) Podaj 3–6 najważniejszych insightów związanych z celem.
3) Zaproponuj 2–4 rekomendacje biznesowe / KPI do monitorowania.
4) Wskaż potencjalne ryzyka / ograniczenia danych.

WAŻNE:
- Zwróć **wyłącznie** JSON zgodny ze schematem (bez lead-in/komentarzy).
- Każdy wpis listy ≤ 180 znaków, bez markdown i emoji.
- Nie wymyślaj liczb — odwołuj się do trendów/ryzyk ogólnych wynikających z profilu.

Format JSON:
{{
  "summary": "string",
  "top_insights": ["string", ...],
  "recommendations": ["string", ...],
  "risks": ["string", ...]
}}"""

# =========================
# Główny generator insightów
# =========================
def generate_insights(
    df: pd.DataFrame,
    goal: Optional[str] = None,
    mode: Literal["standard", "executive", "technical"] = "standard",
    *,
    options: Optional[InsightsOptions] = None,
) -> Dict[str, Any]:
    """
    Generuje insighty AI w stabilnym formacie JSON.
    Zwraca słownik zgodny ze schematem: {summary, top_insights[], recommendations[], risks[]}.
    W razie problemów zwraca minimalny poprawny obiekt + 'raw_text'.
    """
    opts = options or InsightsOptions(mode=mode)
    # Szybkie sanity danych
    if not isinstance(df, pd.DataFrame) or df.empty:
        err = InsightsError(ok=False, code="NO_DATA", message="Brak danych lub pusty DataFrame.", raw_text=None)
        LOGGER.warning(err.message)
        return {
            "summary": "Brak danych do analizy.",
            "top_insights": [],
            "recommendations": [],
            "risks": ["Uzupełnij dane wejściowe."],
        }

    # Profil kontekstu (bez PII)
    context_summary = _profile_context(
        df=df,
        max_cols=max(1, opts.max_profile_cols),
        max_rows=max(1, opts.sample_rows_for_profile),
    )
    user_prompt = _build_user_prompt(context_summary, goal, opts.mode, opts.domain_hint)

    # 1) Pierwsza próba
    raw_output = chat_completion(system=SYSTEM_PROMPT, user=user_prompt, response_format="json")
    parsed = _safe_json_loads(raw_output)
    payload = _coerce_schema(parsed) if parsed is not None else None

    # 2) Retry z hintem, jeżeli nie-JSON lub brak kluczy
    if payload is None:
        retry_prompt = user_prompt + "\n\nUWAGA: Poprzednia odpowiedź naruszała format. Zwróć 100% poprawny JSON — nic poza JSON."
        raw_output_retry = chat_completion(system=SYSTEM_PROMPT, user=retry_prompt, response_format="json")
        parsed_retry = _safe_json_loads(raw_output_retry)
        payload = _coerce_schema(parsed_retry) if parsed_retry is not None else None
        if payload is None:
            # twardy fallback
            LOGGER.error("AI insights: niepoprawny JSON mimo retry.")
            return {
                "summary": "Nie udało się uzyskać poprawnego JSON.",
                "top_insights": [],
                "recommendations": [],
                "risks": [],
                "raw_text": raw_output_retry,
            }
        raw_output = raw_output_retry  # do ewentualnego logowania

    # Przycięcie długości i liczby elementów
    summary = _truncate_str(payload.summary.strip(), 600)
    top_insights = _cap_list_items(payload.top_insights, max_len=180, max_items=8)
    recommendations = _cap_list_items(payload.recommendations, max_len=180, max_items=8)
    risks = _cap_list_items(payload.risks, max_len=180, max_items=8)

    result = {
        "summary": summary,
        "top_insights": top_insights,
        "recommendations": recommendations,
        "risks": risks,
    }
    return result
