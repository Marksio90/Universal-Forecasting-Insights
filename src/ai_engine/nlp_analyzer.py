"""
Text Summarizer PRO - Backend module dla podsumowywania tekstu.

Funkcjonalności:
- 3 strategie: heuristic, AI, hybrid
- TextRank-lite dla heurystyki
- Inteligentne dzielenie zdań z obsługą skrótów
- Detekcja języka (PL/EN)
- Cache dla AI summaries (hash-based)
- Retry logic dla API calls
- Normalizacja i sanityzacja tekstu
- Metrics tracking (reduction, timing)
- Configurable limits
- Comprehensive logging
"""

from __future__ import annotations

import re
import math
import time
import hashlib
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Literal, Dict, Any

# Optional OpenAI integration
try:
    from .openai_integrator import chat_completion
    HAS_AI = True
except Exception:
    chat_completion = None
    HAS_AI = False

# Optional language detection (graceful fallback)
try:
    import importlib
    # Check for the package without a static import so linters won't complain if it's absent
    langdetect_spec = importlib.util.find_spec("langdetect")
    if langdetect_spec is not None:
        lang_module = importlib.import_module("langdetect")
        detect = getattr(lang_module, "detect")
        LangDetectException = getattr(lang_module, "LangDetectException", Exception)
        HAS_LANGDETECT = True
    else:
        HAS_LANGDETECT = False
except Exception:
    HAS_LANGDETECT = False

# ========================================================================================
# KONFIGURACJA
# ========================================================================================

# Default limits
DEFAULT_MAX_CHARS = 300
DEFAULT_MAX_LINES = 3
DEFAULT_MIN_SENTENCE_CHARS = 20
DEFAULT_SENTENCE_LIMIT = 50

# AI configuration
AI_MAX_INPUT_CHARS = 8000
AI_RETRY_COUNT = 2
AI_RETRY_DELAY = 1  # seconds

# Cache TTL (seconds)
CACHE_TTL = 1800  # 30 minutes

# Abbreviations for sentence splitting
COMMON_ABBR = (
    "np", "itd", "itp", "tj", "m.in", "dr", "mgr", "prof", "inż", "lek",
    "ur", "ul", "al", "pl", "os", "mr", "mrs", "ms", "jr", "sr",
    "vs", "etc", "e.g", "i.e", "ca", "no", "fig", "pp", "approx"
)

# Regex patterns (compiled once)
ABBR_PATTERN = "|".join(re.escape(abbr) for abbr in COMMON_ABBR)
MARKDOWN_PATTERN = re.compile(r"(\*\*|\*|__|_|`|#+)")
WHITESPACE_PATTERN = re.compile(r"\s+")
WORD_PATTERN = re.compile(r"\w+", re.UNICODE)

# Sentence splitting regex (improved)
SENTENCE_SPLIT_PATTERN = re.compile(
    rf"""
    (?<!\b(?:{ABBR_PATTERN}))  # Nie dziel po skrótach
    (?<=[\.\!\?\…])             # Po kropce/wykrzykniku/pytajniku
    \s+                         # Spacja/spacje
    """,
    re.IGNORECASE | re.VERBOSE
)

# ========================================================================================
# LOGGING
# ========================================================================================

def get_logger(name: str = "text_summarizer", level: int = logging.INFO) -> logging.Logger:
    """
    Konfiguruje i zwraca logger.
    
    Args:
        name: Nazwa loggera
        level: Poziom logowania
        
    Returns:
        Skonfigurowany logger
    """
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(level)
        handler = logging.StreamHandler()
        handler.setLevel(level)
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
    
    return logger


LOGGER = get_logger()


# ========================================================================================
# DATACLASSES
# ========================================================================================

@dataclass(frozen=True)
class SummarizeOptions:
    """Opcje podsumowania tekstu."""
    max_chars: int = DEFAULT_MAX_CHARS
    max_lines: int = DEFAULT_MAX_LINES
    strategy: Literal["heuristic", "ai", "hybrid"] = "heuristic"
    preserve_language: bool = True
    strip_markdown: bool = False
    normalize_whitespace: bool = True
    min_sentence_chars: int = DEFAULT_MIN_SENTENCE_CHARS
    sentence_limit: int = DEFAULT_SENTENCE_LIMIT
    ai_hard_limits: bool = True
    use_cache: bool = True
    timeout: int = 30


@dataclass
class SummaryMetrics:
    """Metryki podsumowania."""
    original_length: int
    summary_length: int
    reduction_pct: float
    num_sentences_original: int
    num_sentences_summary: int
    strategy_used: str
    time_taken: float
    from_cache: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje do słownika."""
        return {
            "original_length": self.original_length,
            "summary_length": self.summary_length,
            "reduction_pct": self.reduction_pct,
            "num_sentences_original": self.num_sentences_original,
            "num_sentences_summary": self.num_sentences_summary,
            "strategy_used": self.strategy_used,
            "time_taken": self.time_taken,
            "from_cache": self.from_cache
        }


@dataclass
class SummaryResult:
    """Wynik podsumowania."""
    summary: str
    metrics: SummaryMetrics


# ========================================================================================
# CACHE
# ========================================================================================

class SummaryCache:
    """Cache dla podsumowań z TTL."""
    
    def __init__(self, ttl: int = CACHE_TTL):
        self.ttl = ttl
        self._cache: Dict[str, Tuple[str, float]] = {}
    
    def get(self, key: str) -> Optional[str]:
        """Pobiera z cache."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            
            if time.time() - timestamp < self.ttl:
                LOGGER.debug(f"Cache HIT: {key[:16]}...")
                return value
            else:
                del self._cache[key]
                LOGGER.debug(f"Cache EXPIRED: {key[:16]}...")
        
        LOGGER.debug(f"Cache MISS: {key[:16]}...")
        return None
    
    def set(self, key: str, value: str) -> None:
        """Zapisuje do cache."""
        self._cache[key] = (value, time.time())
        LOGGER.debug(f"Cache SET: {key[:16]}...")
    
    def clear(self) -> None:
        """Czyści cache."""
        self._cache.clear()
        LOGGER.info("Summary cache cleared")


# Global cache
_SUMMARY_CACHE = SummaryCache()


def compute_cache_key(text: str, opts: SummarizeOptions) -> str:
    """
    Oblicza klucz cache.
    
    Args:
        text: Tekst do podsumowania
        opts: Opcje
        
    Returns:
        SHA-256 hash
    """
    # Normalize text for consistent hashing
    normalized = text.strip()[:1000]  # First 1000 chars for key
    
    components = [
        normalized,
        str(opts.max_chars),
        str(opts.max_lines),
        opts.strategy,
        str(opts.strip_markdown),
        str(opts.normalize_whitespace)
    ]
    
    key_str = "|".join(components)
    return hashlib.sha256(key_str.encode()).hexdigest()


# ========================================================================================
# TEXT NORMALIZATION
# ========================================================================================

def normalize_text(
    text: str,
    strip_markdown: bool = False,
    compress_whitespace: bool = True
) -> str:
    """
    Normalizuje tekst.
    
    Args:
        text: Tekst do normalizacji
        strip_markdown: Czy usunąć markdown
        compress_whitespace: Czy skompresować whitespace
        
    Returns:
        Znormalizowany tekst
    """
    text = text.strip()
    
    if strip_markdown:
        text = MARKDOWN_PATTERN.sub("", text)
    
    if compress_whitespace:
        text = WHITESPACE_PATTERN.sub(" ", text)
    
    return text


def truncate_at_word_boundary(text: str, max_chars: int) -> str:
    """
    Ucina tekst na granicy słowa.
    
    Args:
        text: Tekst do ucięcia
        max_chars: Maksymalna długość
        
    Returns:
        Ucięty tekst
    """
    if max_chars <= 0:
        return ""
    
    if len(text) <= max_chars:
        return text
    
    # Ucina do max_chars
    truncated = text[:max_chars].rstrip()
    
    # Znajdź ostatni separator słowa
    match = re.search(r"[\s\-—,;:]", truncated[::-1])
    
    if match:
        truncated = truncated[:len(truncated) - match.start()]
    
    # Usuń trailing punctuation
    truncated = truncated.rstrip(" .,;:-—")
    
    return truncated + "…"


def safe_truncate(text: str, max_chars: int) -> str:
    """
    Bezpieczne ucięcie tekstu.
    
    Args:
        text: Tekst
        max_chars: Max długość
        
    Returns:
        Ucięty tekst
    """
    return truncate_at_word_boundary(text.strip(), max_chars)


# ========================================================================================
# SENTENCE SPLITTING
# ========================================================================================

def split_into_sentences(text: str, limit: int = DEFAULT_SENTENCE_LIMIT) -> List[str]:
    """
    Dzieli tekst na zdania z obsługą skrótów.
    
    Args:
        text: Tekst do podziału
        limit: Maksymalna liczba zdań
        
    Returns:
        Lista zdań
    """
    # Zabezpieczenie kropek w liczbach (3.14 -> 3§DOT§14)
    text = re.sub(r"(\d)\.(\d)", r"\1§DOT§\2", text)
    
    # Zabezpieczenie inicjałów (A.B. -> A§INIT§B§INIT§)
    text = re.sub(r"([A-Za-z])\.([A-Za-z])\.", r"\1§INIT§\2§INIT§", text)
    
    # Split
    parts = SENTENCE_SPLIT_PATTERN.split(text)
    
    # Przywróć kropki
    parts = [
        p.replace("§DOT§", ".").replace("§INIT§", ".")
        for p in parts
    ]
    
    # Czyszczenie
    sentences = [
        p.strip()
        for p in parts
        if p and p.strip()
    ]
    
    # Limit
    if len(sentences) > limit:
        LOGGER.debug(f"Limiting sentences from {len(sentences)} to {limit}")
        sentences = sentences[:limit]
    
    return sentences


def count_sentences(text: str) -> int:
    """
    Liczy zdania w tekście.
    
    Args:
        text: Tekst
        
    Returns:
        Liczba zdań
    """
    sentences = split_into_sentences(text, limit=1000)
    return len(sentences)


# ========================================================================================
# LANGUAGE DETECTION
# ========================================================================================

def detect_language(text: str) -> str:
    """
    Wykrywa język tekstu (PL/EN).
    
    Args:
        text: Tekst do analizy
        
    Returns:
        Kod języka ('pl' lub 'en')
    """
    # Jeśli mamy langdetect - użyj go
    if HAS_LANGDETECT:
        try:
            lang = detect(text[:1000])  # First 1000 chars
            
            if lang in ("pl", "en"):
                LOGGER.debug(f"Detected language: {lang}")
                return lang
            
            # Fallback to EN for other languages
            return "en"
            
        except Exception as e:
            LOGGER.warning(f"Language detection failed: {e}")
    
    # Fallback: prosty heurystyczny detektor
    polish_chars = re.findall(r"[ąćęłńóśźż]", text.lower())
    
    if len(polish_chars) > 5:  # Arbitrary threshold
        return "pl"
    
    return "en"


# ========================================================================================
# HEURISTIC SUMMARIZATION (TextRank-lite)
# ========================================================================================

def tokenize_sentence(sentence: str) -> List[str]:
    """
    Tokenizuje zdanie na słowa.
    
    Args:
        sentence: Zdanie
        
    Returns:
        Lista tokenów
    """
    tokens = WORD_PATTERN.findall(sentence.lower())
    
    # Filter short tokens and numbers
    return [
        t for t in tokens
        if len(t) > 2 and not t.isdigit()
    ]


def score_sentences_tfidf(sentences: List[str]) -> List[float]:
    """
    Punktuje zdania używając TF-IDF-like scoring.
    
    Args:
        sentences: Lista zdań
        
    Returns:
        Lista scores
    """
    # Tokenize all sentences
    docs = [tokenize_sentence(s) for s in sentences]
    
    # Compute document frequency (DF)
    df: Dict[str, int] = {}
    term_freq: List[Dict[str, int]] = []
    
    for doc in docs:
        # Term frequency for this doc
        counts: Dict[str, int] = {}
        for word in doc:
            counts[word] = counts.get(word, 0) + 1
        
        term_freq.append(counts)
        
        # Document frequency
        for word in set(doc):
            df[word] = df.get(word, 0) + 1
    
    # Compute scores
    n_docs = max(1, len(sentences))
    scores: List[float] = []
    
    for i, counts in enumerate(term_freq):
        score = 0.0
        
        for word, freq in counts.items():
            # IDF
            idf = math.log((n_docs + 1) / (1 + df.get(word, 1))) + 1.0
            score += freq * idf
        
        # Length penalty (prefer shorter sentences for readability)
        length_penalty = 1.0 / (1.0 + max(0, len(sentences[i]) - 180) / 180.0)
        
        scores.append(score * length_penalty)
    
    return scores


def heuristic_summary(text: str, opts: SummarizeOptions) -> str:
    """
    Generuje podsumowanie heurystyczne (TextRank-lite).
    
    Args:
        text: Tekst do podsumowania
        opts: Opcje
        
    Returns:
        Podsumowanie
    """
    # Split sentences
    sentences = split_into_sentences(text, limit=opts.sentence_limit)
    
    # Filter too short sentences
    sentences = [
        s for s in sentences
        if len(s) >= opts.min_sentence_chars
    ]
    
    if not sentences:
        return safe_truncate(text, opts.max_chars)
    
    # Score sentences
    scores = score_sentences_tfidf(sentences)
    
    # Select top sentences (preserve original order)
    ranked_indices = sorted(
        range(len(sentences)),
        key=lambda i: scores[i],
        reverse=True
    )
    
    # Take top N (2x max_lines for selection)
    top_n = max(1, opts.max_lines * 2)
    selected_indices = set(ranked_indices[:top_n])
    
    chosen_sentences = [
        s for i, s in enumerate(sentences)
        if i in selected_indices
    ]
    
    # Compose summary within limits
    lines: List[str] = []
    total_chars = 0
    
    for sentence in chosen_sentences:
        if len(lines) >= opts.max_lines:
            break
        
        # Truncate if needed
        if total_chars + len(sentence) > opts.max_chars:
            sentence = truncate_at_word_boundary(
                sentence,
                opts.max_chars - total_chars
            )
        
        if sentence:
            lines.append(sentence)
            total_chars += len(sentence) + 1  # +1 for space
    
    if not lines:
        return safe_truncate(text, opts.max_chars)
    
    summary = " ".join(lines)
    
    return enforce_final_limits(summary, opts.max_chars, opts.max_lines)


# ========================================================================================
# AI SUMMARIZATION
# ========================================================================================

def ai_summary_with_retry(text: str, opts: SummarizeOptions) -> str:
    """
    Generuje podsumowanie AI z retry logic.
    
    Args:
        text: Tekst do podsumowania
        opts: Opcje
        
    Returns:
        Podsumowanie
        
    Raises:
        RuntimeError: Jeśli AI niedostępne
        Exception: Po wyczerpaniu retry
    """
    if not HAS_AI or chat_completion is None:
        raise RuntimeError("AI integration not available")
    
    # Detect language
    lang = detect_language(text) if opts.preserve_language else "en"
    
    # System prompt
    system_prompt = (
        "You are a world-class summarizer. Return ONLY the summary text without extra commentary."
        if lang == "en" else
        "Jesteś światowej klasy streszczaczem. Zwróć WYŁĄCZNIE streszczenie, bez komentarzy."
    )
    
    # Hard limits hint
    hard_limits = ""
    if opts.ai_hard_limits:
        hard_limits = (
            f"(HARD LIMITS: ≤ {opts.max_lines} lines, ≤ {opts.max_chars} characters)\n"
        )
    
    # Truncate input to safe size
    text_input = text.strip()
    if len(text_input) > AI_MAX_INPUT_CHARS:
        text_input = text_input[:AI_MAX_INPUT_CHARS]
        LOGGER.debug(f"Truncated input to {AI_MAX_INPUT_CHARS} chars")
    
    # User prompt
    if lang == "en":
        user_prompt = (
            f"Summarize the text in at most {opts.max_lines} sentences "
            f"and {opts.max_chars} characters.\n"
            f"{hard_limits}\n"
            f"Text:\n{text_input}"
        )
    else:
        user_prompt = (
            f"Streść tekst w maksymalnie {opts.max_lines} zdaniach "
            f"i {opts.max_chars} znakach.\n"
            f"{hard_limits}\n"
            f"Tekst:\n{text_input}"
        )
    
    # Retry loop
    last_error = None
    
    for attempt in range(AI_RETRY_COUNT):
        try:
            LOGGER.debug(f"AI summary attempt {attempt + 1}/{AI_RETRY_COUNT}")
            
            response = chat_completion(system=system_prompt, user=user_prompt)
            
            if not isinstance(response, str):
                raise ValueError("AI returned non-string response")
            
            summary = response.strip()
            
            if len(summary) < 10:
                raise ValueError("AI response too short")
            
            LOGGER.debug(f"AI summary successful on attempt {attempt + 1}")
            return summary
            
        except Exception as e:
            last_error = e
            LOGGER.warning(f"AI summary failed (attempt {attempt + 1}): {e}")
            
            if attempt < AI_RETRY_COUNT - 1:
                time.sleep(AI_RETRY_DELAY)
    
    # All retries failed
    raise Exception(f"AI summary failed after {AI_RETRY_COUNT} attempts: {last_error}")


# ========================================================================================
# FINAL LIMITS ENFORCEMENT
# ========================================================================================

def enforce_final_limits(text: str, max_chars: int, max_lines: int) -> str:
    """
    Wymusza finalne limity na tekście.
    
    Args:
        text: Tekst
        max_chars: Max znaki
        max_lines: Max linie
        
    Returns:
        Ograniczony tekst
    """
    # Split into lines/sentences
    parts = re.split(r"\s*(?:\n|(?<=[.!?])\s{2,})\s*", text.strip())
    
    lines: List[str] = []
    
    for part in parts:
        if not part:
            continue
        
        if len(lines) >= max_lines:
            break
        
        lines.append(part.strip())
    
    # Join and truncate
    result = " ".join(lines)
    
    return safe_truncate(result, max_chars)


# ========================================================================================
# MAIN API
# ========================================================================================

def summarize_text(
    text: str,
    max_chars: int = DEFAULT_MAX_CHARS,
    max_lines: int = DEFAULT_MAX_LINES,
    use_ai: bool = False
) -> str:
    """
    Podsumowuje tekst (backward compatible API).
    
    Args:
        text: Tekst do podsumowania
        max_chars: Maksymalna długość podsumowania
        max_lines: Maksymalna liczba linii
        use_ai: Czy użyć AI (hybrid strategy)
        
    Returns:
        Podsumowanie
    """
    opts = SummarizeOptions(
        max_chars=max_chars,
        max_lines=max_lines,
        strategy="hybrid" if use_ai else "heuristic"
    )
    
    result = summarize_text_pro(text, opts)
    return result.summary


def summarize_text_pro(
    text: str,
    opts: SummarizeOptions = SummarizeOptions()
) -> SummaryResult:
    """
    Podsumowuje tekst z pełnymi opcjami i metrykami.
    
    Args:
        text: Tekst do podsumowania
        opts: Opcje podsumowania
        
    Returns:
        SummaryResult z podsumowaniem i metrykami
    """
    start_time = time.time()
    
    # Validation
    if not isinstance(text, str) or not text.strip():
        empty_metrics = SummaryMetrics(
            original_length=0,
            summary_length=0,
            reduction_pct=0.0,
            num_sentences_original=0,
            num_sentences_summary=0,
            strategy_used="none",
            time_taken=0.0
        )
        return SummaryResult(summary="", metrics=empty_metrics)
    
    # Normalize
    normalized = normalize_text(
        text,
        strip_markdown=opts.strip_markdown,
        compress_whitespace=opts.normalize_whitespace
    )
    
    if not normalized:
        empty_metrics = SummaryMetrics(
            original_length=len(text),
            summary_length=0,
            reduction_pct=100.0,
            num_sentences_original=0,
            num_sentences_summary=0,
            strategy_used="none",
            time_taken=time.time() - start_time
        )
        return SummaryResult(summary="", metrics=empty_metrics)
    
    # Check cache
    from_cache = False
    summary = None
    
    if opts.use_cache:
        cache_key = compute_cache_key(normalized, opts)
        summary = _SUMMARY_CACHE.get(cache_key)
        
        if summary is not None:
            from_cache = True
            LOGGER.info("Returning cached summary")
    
    # Generate summary if not cached
    if summary is None:
        try:
            if opts.strategy == "ai":
                # AI only
                summary = ai_summary_with_retry(normalized, opts)
                summary = enforce_final_limits(summary, opts.max_chars, opts.max_lines)
                
            elif opts.strategy == "hybrid":
                # Try AI, fallback to heuristic
                if HAS_AI and chat_completion is not None:
                    try:
                        summary = ai_summary_with_retry(normalized, opts)
                        summary = enforce_final_limits(summary, opts.max_chars, opts.max_lines)
                    except Exception as e:
                        LOGGER.warning(f"AI failed, using heuristic fallback: {e}")
                        summary = heuristic_summary(normalized, opts)
                else:
                    LOGGER.warning("AI not available, using heuristic")
                    summary = heuristic_summary(normalized, opts)
                    
            else:
                # Heuristic only
                summary = heuristic_summary(normalized, opts)
            
            # Cache result
            if opts.use_cache and summary:
                _SUMMARY_CACHE.set(cache_key, summary)
                
        except Exception as e:
            LOGGER.exception("Summary generation failed, using safe truncate")
            summary = safe_truncate(normalized, opts.max_chars)
    
    # Compute metrics
    elapsed = time.time() - start_time
    
    original_sentences = count_sentences(normalized)
    summary_sentences = count_sentences(summary)
    
    reduction_pct = (
        (1.0 - len(summary) / len(normalized)) * 100.0
        if len(normalized) > 0
        else 0.0
    )
    
    metrics = SummaryMetrics(
        original_length=len(normalized),
        summary_length=len(summary),
        reduction_pct=reduction_pct,
        num_sentences_original=original_sentences,
        num_sentences_summary=summary_sentences,
        strategy_used=opts.strategy,
        time_taken=elapsed,
        from_cache=from_cache
    )
    
    LOGGER.info(
        f"Summary generated: {metrics.original_length} -> {metrics.summary_length} chars "
        f"({metrics.reduction_pct:.1f}% reduction) in {elapsed:.3f}s"
    )
    
    return SummaryResult(summary=summary, metrics=metrics)


# ========================================================================================
# UTILITIES
# ========================================================================================

def clear_summary_cache() -> None:
    """Czyści cache podsumowań."""
    _SUMMARY_CACHE.clear()


def get_cache_stats() -> Dict[str, Any]:
    """
    Zwraca statystyki cache.
    
    Returns:
        Słownik ze statystykami
    """
    return {
        "size": len(_SUMMARY_CACHE._cache),
        "ttl": _SUMMARY_CACHE.ttl
    }


def test_summarization(text: str) -> Dict[str, Any]:
    """
    Testuje wszystkie strategie podsumowania.
    
    Args:
        text: Tekst testowy
        
    Returns:
        Słownik z wynikami dla każdej strategii
    """
    results = {}
    
    for strategy in ["heuristic", "ai", "hybrid"]:
        opts = SummarizeOptions(strategy=strategy, use_cache=False)  # type: ignore
        
        try:
            result = summarize_text_pro(text, opts)
            results[strategy] = {
                "summary": result.summary,
                "metrics": result.metrics.to_dict()
            }
        except Exception as e:
            results[strategy] = {
                "error": str(e)
            }
    
    return results