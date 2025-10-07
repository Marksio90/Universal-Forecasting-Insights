from __future__ import annotations
import re
from typing import Optional

try:
    # Optional: użycie Twojego integratora, jeśli istnieje
    from .openai_integrator import chat_completion
except ImportError:
    chat_completion = None


def summarize_text(
    text: str,
    max_chars: int = 300,
    max_lines: int = 3,
    use_ai: bool = False,
) -> str:
    """
    Tworzy krótkie streszczenie tekstu.
    - Czyści puste linie, zbędne spacje
    - Skraca każdy akapit
    - Opcjonalnie używa AI do generowania podsumowania (jeśli dostępny openai_integrator)
    """
    if not text or not isinstance(text, str):
        return ""

    # ---- Preprocessing
    cleaned = re.sub(r"\s+", " ", text.strip())
    if not cleaned:
        return ""

    # Jeśli AI do streszczeń
    if use_ai and chat_completion is not None:
        try:
            system = "Jesteś asystentem, który streszcza tekst zwięźle i rzeczowo (max 500 znaków)."
            prompt = f"Streść poniższy tekst w maks. 500 znakach:\n\n{cleaned[:3000]}"
            summary = chat_completion(system=system, user=prompt)
            if isinstance(summary, str) and len(summary.strip()) > 10:
                return summary.strip()
        except Exception:
            pass  # fallback poniżej

    # ---- Manual summary (heurystyczny)
    # Podział na zdania
    parts = re.split(r"(?<=[.!?])\s+", cleaned)
    nonempty = [p.strip() for p in parts if len(p.strip()) > 0]

    if not nonempty:
        return cleaned[:max_chars] + ("..." if len(cleaned) > max_chars else "")

    # 1️⃣ Wersja krótsza niż limit → zwróć całość
    joined = " ".join(nonempty[:max_lines])
    if len(joined) <= max_chars:
        return joined

    # 2️⃣ Inaczej – zbuduj mini-lead
    short_lines = []
    total_len = 0
    for ln in nonempty:
        if total_len + len(ln) > max_chars:
            break
        short_lines.append(ln)
        total_len += len(ln)
        if len(short_lines) >= max_lines:
            break

    result = " ".join(short_lines)
    return result + ("..." if len(result) < len(cleaned) else "")
