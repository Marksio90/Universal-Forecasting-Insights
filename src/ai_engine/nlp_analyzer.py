# text_summarizer.py — TURBO PRO
from __future__ import annotations

import re
import math
import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple, Literal

try:
    # Opcjonalna integracja z Twoim wrapperem OpenAI
    from .openai_integrator import chat_completion  # type: ignore
except Exception:  # pragma: no cover
    chat_completion = None  # type: ignore


# =========================
# Logger
# =========================
def _get_logger(name: str = "text_summarizer", level: int = logging.INFO) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(level)
        h = logging.StreamHandler()
        h.setLevel(level)
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s", "%Y-%m-%d %H:%M:%S"))
        log.addHandler(h)
        log.propagate = False
    return log

LOGGER = _get_logger()


# =========================
# Konfiguracja / typy
# =========================
@dataclass(frozen=True)
class SummarizeOptions:
    max_chars: int = 300
    max_lines: int = 3
    strategy: Literal["heuristic", "ai", "hybrid"] = "heuristic"
    preserve_language: bool = True       # AI ma pisać w języku wejścia
    strip_markdown: bool = False         # usuń **, _, # jeśli chcesz
    normalize_whitespace: bool = True    # kompresja spacji i łamań
    min_sentence_chars: int = 20         # ignoruj „zdania” bardzo krótkie
    sentence_limit: int = 50             # maks. zdań do rozważenia (wydajność)
    ai_hard_limits: bool = True          # AI dostaje twarde limity (chars/lines) w prompt


# =========================
# API (główna funkcja)
# =========================
def summarize_text(
    text: str,
    max_chars: int = 300,
    max_lines: int = 3,
    use_ai: bool = False,
) -> str:
    """
    Wstecznie kompatybilne API:
    - `use_ai=False`  -> czysto heurystyczny skrót
    - `use_ai=True`   -> tryb hybrydowy (AI + twardy fallback)
    """
    opts = SummarizeOptions(
        max_chars=max_chars,
        max_lines=max_lines,
        strategy="hybrid" if use_ai else "heuristic",
    )
    return summarize_text_pro(text, opts)


# =========================
# Public: wersja PRO z opcjami
# =========================
def summarize_text_pro(text: str, opts: SummarizeOptions = SummarizeOptions()) -> str:
    if not isinstance(text, str) or not text.strip():
        return ""

    # 1) Normalize
    norm = _normalize(text, strip_md=opts.strip_markdown, compress=opts.normalize_whitespace)
    if not norm:
        return ""

    # 2) Strategia
    try:
        if opts.strategy == "ai":
            ai_out = _ai_summary(norm, opts)
            return _enforce_final_limits(ai_out, opts.max_chars, opts.max_lines)
        elif opts.strategy == "hybrid":
            if chat_completion is not None:
                try:
                    ai_out = _ai_summary(norm, opts)
                    if _is_reasonable(ai_out):
                        return _enforce_final_limits(ai_out, opts.max_chars, opts.max_lines)
                except Exception as e:  # AI nie wyszło, lecimy dalej
                    LOGGER.warning("AI summary failed, fallback to heuristic: %s", e)
            # fallback → heuristic
            return _heuristic_summary(norm, opts)
        else:
            # heuristic
            return _heuristic_summary(norm, opts)
    except Exception as e:
        LOGGER.exception("summarize_text_pro failed; returning safe fallback")
        return _safe_truncate(norm, opts.max_chars)


# =========================
# Heurystyka: split → rank → compose
# =========================
_ABBR = (
    r"np|itd|itp|tj|m\.in|dr|mgr|prof|inz|lek|ur|ul|al|pl|os|mr|mrs|ms|dr|jr|sr|vs|etc|e\.g|i\.e|ca|no|fig|pp"
)

_SENT_SPLIT_REGEX = re.compile(
    rf"""
    (?<!\b(?:{_ABBR})   # nie dziel po skrótach typu 'np.' 'itd.' 'dr.' itd.
       )                 # koniec negatywnego lookbehind (variable len via trick — używamy heurystyki)
    (?<=[\.\!\?\…])      # po kropce/wykrzykniku/znaku zapyt.
    \s+                  # spacja/y
    """,
    re.IGNORECASE | re.VERBOSE,
)

def _split_sentences(text: str, limit: int) -> List[str]:
    # Szybkie zabezpieczenia: kropki w liczbach / inicjałach
    t = re.sub(r"(\d)\.(\d)", r"\1§DOT§\2", text)  # 3.14 -> 3§DOT§14
    t = re.sub(r"([A-Za-z])\.([A-Za-z])\.", r"\1§INIT§\2§INIT§", t)  # A.B. -> A§INIT§B§INIT§
    parts = _SENT_SPLIT_REGEX.split(t)
    # Przywróć kropki
    parts = [p.replace("§DOT§", ".").replace("§INIT§", ".") for p in parts]
    # Czyszczenie
    sents = [p.strip(" \n\t") for p in parts if p and len(p.strip()) > 0]
    # Limit
    if len(sents) > limit:
        sents = sents[:limit]
    return sents

def _heuristic_summary(text: str, opts: SummarizeOptions) -> str:
    sents = _split_sentences(text, limit=opts.sentence_limit)
    sents = [s for s in sents if len(s) >= opts.min_sentence_chars]
    if not sents:
        return _safe_truncate(text, opts.max_chars)

    # „TextRank-lite”: punktuj zdania po tf-idf-ish (bez zewnętrznych zależności)
    scores = _score_sentences(sents)
    # wybierz najlepsze zdania zachowując kolejność oryginalną
    ranked_idx = sorted(range(len(sents)), key=lambda i: scores[i], reverse=True)
    pick = set(ranked_idx[: max(1, opts.max_lines * 2)])  # weź kilka najlepszych, potem potnij do limitów
    chosen = [s for i, s in enumerate(sents) if i in pick]

    # Sklejaj do limitów z estetycznym domknięciem
    out_lines: List[str] = []
    total = 0
    for s in chosen:
        if len(out_lines) >= opts.max_lines:
            break
        if total + len(s) > opts.max_chars:
            # spróbuj przyciąć zdanie na granicy słowa
            s = _truncate_at_word(s, opts.max_chars - total)
        if s:
            out_lines.append(s)
            total += len(s) + 1

    if not out_lines:
        return _safe_truncate(text, opts.max_chars)

    out = " ".join(out_lines)
    return _enforce_final_limits(out, opts.max_chars, opts.max_lines)

_WORD = re.compile(r"\w+", re.UNICODE)

def _score_sentences(sents: List[str]) -> List[float]:
    # prosty tf-idf-ish: słowa -> wagi; ignoruj krótkie/numeryczne
    def tokenize(s: str) -> List[str]:
        toks = [t.lower() for t in _WORD.findall(s)]
        return [t for t in toks if len(t) > 2 and not t.isdigit()]

    docs = [tokenize(s) for s in sents]
    df: dict[str, int] = {}
    tf: List[dict[str, int]] = []
    for doc in docs:
        counts: dict[str, int] = {}
        for w in doc:
            counts[w] = counts.get(w, 0) + 1
        tf.append(counts)
        for w in set(doc):
            df[w] = df.get(w, 0) + 1

    n_docs = max(1, len(sents))
    scores: List[float] = []
    for i, counts in enumerate(tf):
        sc = 0.0
        for w, c in counts.items():
            idf = math.log((n_docs + 1) / (1 + df.get(w, 1))) + 1.0
            sc += c * idf
        # preferuj zdania krótsze (czytelność)
        length_penalty = 1.0 / (1.0 + max(0, len(sents[i]) - 180) / 180.0)
        scores.append(sc * length_penalty)
    return scores


# =========================
# AI Summary
# =========================
def _detect_language_sample(s: str) -> str:
    # bardzo prosty heurystyczny detektor (PL vs EN) do promptu
    if re.search(r"[ąćęłńóśźż]", s.lower()):
        return "pl"
    return "en"

def _ai_summary(text: str, opts: SummarizeOptions) -> str:
    if chat_completion is None:
        raise RuntimeError("chat_completion integrator not available")
    lang = _detect_language_sample(text) if opts.preserve_language else "en"
    sys = (
        "You are a world-class summarizer. Return ONLY the summary text without extra commentary.\n"
        if lang == "en" else
        "Jesteś światowej klasy streszczaczem. Zwróć WYŁĄCZNIE streszczenie, bez komentarzy."
    )
    hard = (
        f"(HARD LIMITS: ≤ {opts.max_lines} lines, ≤ {opts.max_chars} characters total)\n"
        if opts.ai_hard_limits else ""
    )
    # skróć wejście do bezpiecznego rozmiaru dla kontekstu
    text_in = text.strip()
    if len(text_in) > 8000:
        text_in = text_in[:8000]

    prompt = (
        (f"Summarize the text in at most {opts.max_lines} sentences and {opts.max_chars} characters.\n{hard}\nText:\n{text_in}")
        if lang == "en"
        else (f"Streść tekst w maksymalnie {opts.max_lines} zdaniach i {opts.max_chars} znakach.\n{hard}\nTekst:\n{text_in}")
    )
    out = chat_completion(system=sys, user=prompt)  # type: ignore
    if not isinstance(out, str):
        raise ValueError("AI returned non-string")
    return out.strip()


# =========================
# Normalizacja i limity
# =========================
_MD = re.compile(r"(\*\*|\*|__|_|`|#+)")
_WS = re.compile(r"\s+")

def _normalize(text: str, *, strip_md: bool, compress: bool) -> str:
    t = text.strip()
    if strip_md:
        t = _MD.sub("", t)
    if compress:
        t = _WS.sub(" ", t)
    return t

def _truncate_at_word(s: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(s) <= max_chars:
        return s
    cut = s[:max_chars].rstrip()
    # spróbuj obciąć do najbliższego separatora słowa
    m = re.search(r"[ \-\—\,\;\:]", cut[::-1])
    if m:
        cut = cut[: len(cut) - m.start()]
    return cut.rstrip(" .,;:-—") + "…"

def _safe_truncate(text: str, max_chars: int) -> str:
    t = text.strip()
    return _truncate_at_word(t, max_chars)

def _is_reasonable(s: str) -> bool:
    return isinstance(s, str) and len(s.strip()) >= 10

def _enforce_final_limits(s: str, max_chars: int, max_lines: int) -> str:
    # ogranicz liczbę linii
    parts = re.split(r"\s*(?:\n|(?<=[.!?])\s{2,})\s*", s.strip())
    lines: List[str] = []
    for p in parts:
        if not p:
            continue
        if len(lines) >= max_lines:
            break
        lines.append(p.strip())
    out = " ".join(lines)
    return _safe_truncate(out, max_chars)
