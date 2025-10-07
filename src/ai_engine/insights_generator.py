from __future__ import annotations
import json
import pandas as pd
from typing import Any
from .openai_integrator import chat_completion

SYSTEM_PROMPT = (
    "Jesteś seniorem Data Scientist i Analitykiem Biznesowym. "
    "Analizujesz dane i cele użytkownika. "
    "Zwróć wyniki w ściśle poprawnym JSON z kluczami: "
    "`summary`, `top_insights`, `recommendations`, `risks`."
)

# -------------------------------
# Pomocnicze funkcje
# -------------------------------
def _profile_context(df: pd.DataFrame, max_cols: int = 20) -> str:
    """Tworzy skrótowy opis struktury danych do promptu."""
    info_lines = []
    info_lines.append(f"- Wiersze: {len(df):,}")
    info_lines.append(f"- Kolumny: {df.shape[1]:,}")

    missing_pct = float((df.isna().sum().sum()) / (df.size or 1)) * 100
    dupes = int(df.duplicated().sum())
    info_lines.append(f"- Braki: {missing_pct:.2f}%")
    info_lines.append(f"- Duplikaty: {dupes:,}")

    # Typy kolumn (zagregowane)
    type_counts = df.dtypes.astype(str).value_counts().to_dict()
    info_lines.append("- Typy kolumn: " + ", ".join([f"{t}: {n}" for t, n in type_counts.items()]))

    # Najczęstsze kolumny z brakami
    miss_cols = df.isna().mean().sort_values(ascending=False).head(5)
    if miss_cols.max() > 0:
        info_lines.append("- Najwięcej braków: " + ", ".join([f"{c} ({p:.1%})" for c, p in miss_cols.items()]))

    # Unikalność
    nunique_cols = df.nunique().sort_values(ascending=False).head(5)
    info_lines.append("- Top unikalnych wartości: " + ", ".join([f"{c}: {n}" for c, n in nunique_cols.items()]))

    # Wariantywność (jeśli numeric)
    num_cols = df.select_dtypes("number").nunique().sort_values(ascending=False).head(5)
    if not num_cols.empty:
        info_lines.append("- Top zmiennych numerycznych: " + ", ".join([f"{c}: {n}" for c, n in num_cols.items()]))

    return "\n".join(info_lines[:max_cols])

# -------------------------------
# Główny generator insightów
# -------------------------------
def generate_insights(df: pd.DataFrame, goal: str | None = None, mode: str = "standard") -> dict[str, Any]:
    """
    Generuje insighty AI (JSON) na bazie danych i celu.
    Zwraca dict z kluczami: summary, top_insights, recommendations, risks.
    """
    goal_txt = goal or "Brak sprecyzowanego celu."
    context_summary = _profile_context(df)

    user_prompt = f"""Dane (skrótowy opis):
{context_summary}

Cel użytkownika:
{goal_txt}

Zadanie:
1) Stwórz krótkie podsumowanie danych i ich potencjału (executive summary).
2) Wypisz 3–6 najważniejszych insightów dotyczących celu.
3) Zaproponuj 2–4 rekomendacje biznesowe / KPI do monitorowania.
4) Wypisz potencjalne ryzyka, anomalie lub ograniczenia danych.

Zwróć wynik **tylko jako JSON** w formacie:
{{
  "summary": "...",
  "top_insights": ["...", "..."],
  "recommendations": ["...", "..."],
  "risks": ["...", "..."]
}}"""

    # Wywołanie OpenAI
    raw_output = chat_completion(system=SYSTEM_PROMPT, user=user_prompt, response_format="json")

    # Próba parsowania JSON
    try:
        parsed = json.loads(raw_output)
        # sanity check
        if isinstance(parsed, dict) and "summary" in parsed:
            return parsed
        else:
            return {"summary": "Nie udało się sparsować pełnej struktury.", "raw_text": raw_output}
    except Exception:
        # fallback
        return {
            "summary": "Nie udało się sparsować odpowiedzi AI.",
            "raw_text": raw_output,
        }
