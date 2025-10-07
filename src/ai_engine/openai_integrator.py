"""
OpenAI Integrator PRO++ - Backend module dla integracji z OpenAI API.

Funkcjonalności:
- Chat completions z retry logic i exponential backoff
- JSON response format enforcement
- Rate limiting (token bucket algorithm)
- Request caching z TTL
- Cost tracking (tokens i USD)
- Circuit breaker pattern
- Comprehensive logging i metrics
- Streaming support z callbacks
- Multiple API key support
- Configurable timeouts i retries
- Error recovery strategies
"""

from __future__ import annotations

import os
import re
import json
import time
import math
import random
import hashlib
import logging
import threading
from dataclasses import dataclass, field
from typing import Optional, Callable, Any, Dict, List, Literal, Union, Tuple
from collections import defaultdict

# Optional Streamlit integration
try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    st = None
    HAS_STREAMLIT = False

# OpenAI SDK
try:
    from openai import OpenAI
    from openai import (
        APIError,
        RateLimitError,
        APITimeoutError,
        APIConnectionError,
        BadRequestError,
        AuthenticationError
    )
    HAS_OPENAI = True
except ImportError:
    # Fallback types
    OpenAI = object  # type: ignore
    
    class APIError(Exception):
        pass
    
    class RateLimitError(APIError):
        pass
    
    class APITimeoutError(APIError):
        pass
    
    class APIConnectionError(APIError):
        pass
    
    class BadRequestError(APIError):
        pass
    
    class AuthenticationError(APIError):
        pass
    
    HAS_OPENAI = False

# ========================================================================================
# KONFIGURACJA
# ========================================================================================

# Default model
DEFAULT_MODEL = "gpt-4o-mini"

# Model aliases
MODEL_ALIASES = {
    "gpt-4o-mini": "gpt-4o-mini",
    "gpt-4o": "gpt-4o",
    "gpt-4-turbo": "gpt-4-turbo-preview",
    "gpt-4": "gpt-4",
    "gpt-3.5-turbo": "gpt-3.5-turbo",
}

# Pricing per 1M tokens (USD) - update as needed
TOKEN_PRICING = {
    "gpt-4o-mini": {"input": 0.15, "output": 0.60},
    "gpt-4o": {"input": 2.50, "output": 10.00},
    "gpt-4-turbo-preview": {"input": 10.00, "output": 30.00},
    "gpt-4": {"input": 30.00, "output": 60.00},
    "gpt-3.5-turbo": {"input": 0.50, "output": 1.50},
}

# Retry configuration
DEFAULT_RETRIES = 3
DEFAULT_TIMEOUT = 60
DEFAULT_BASE_BACKOFF = 1.0
MAX_BACKOFF = 16.0

# Rate limiting (requests per minute)
RATE_LIMIT_RPM = 60
RATE_LIMIT_WINDOW = 60  # seconds

# Cache TTL
CACHE_TTL = 1800  # 30 minutes

# Circuit breaker
CIRCUIT_BREAKER_THRESHOLD = 5  # failures before opening
CIRCUIT_BREAKER_TIMEOUT = 60  # seconds before retry

# ========================================================================================
# LOGGING
# ========================================================================================

def get_logger(name: str = "openai_integrator", level: int = logging.INFO) -> logging.Logger:
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

@dataclass
class OpenAIConfig:
    """Konfiguracja OpenAI."""
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    project: Optional[str] = None
    model: str = DEFAULT_MODEL
    temperature: float = 0.2
    max_tokens: int = 1000
    timeout: int = DEFAULT_TIMEOUT
    retries: int = DEFAULT_RETRIES


@dataclass
class CompletionMetrics:
    """Metryki completion."""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    cost_usd: float
    duration_seconds: float
    model: str
    cached: bool = False
    retries: int = 0


@dataclass
class CompletionResult:
    """Wynik completion."""
    content: str
    metrics: CompletionMetrics
    
    def to_dict(self) -> Dict[str, Any]:
        """Konwertuje do słownika."""
        return {
            "content": self.content,
            "metrics": {
                "prompt_tokens": self.metrics.prompt_tokens,
                "completion_tokens": self.metrics.completion_tokens,
                "total_tokens": self.metrics.total_tokens,
                "cost_usd": self.metrics.cost_usd,
                "duration_seconds": self.metrics.duration_seconds,
                "model": self.metrics.model,
                "cached": self.metrics.cached,
                "retries": self.metrics.retries
            }
        }


# ========================================================================================
# RATE LIMITER
# ========================================================================================

class TokenBucketRateLimiter:
    """Token bucket rate limiter."""
    
    def __init__(self, rate: int, window: int):
        """
        Initialize rate limiter.
        
        Args:
            rate: Max requests per window
            window: Time window in seconds
        """
        self.rate = rate
        self.window = window
        self.tokens = rate
        self.last_update = time.time()
        self.lock = threading.Lock()
    
    def acquire(self, blocking: bool = True, timeout: Optional[float] = None) -> bool:
        """
        Acquire permission to make request.
        
        Args:
            blocking: Whether to block if no tokens available
            timeout: Max time to wait (None = infinite)
            
        Returns:
            True if acquired, False otherwise
        """
        start_time = time.time()
        
        while True:
            with self.lock:
                now = time.time()
                elapsed = now - self.last_update
                
                # Refill tokens
                self.tokens = min(
                    self.rate,
                    self.tokens + (elapsed * self.rate / self.window)
                )
                self.last_update = now
                
                # Check if we can proceed
                if self.tokens >= 1:
                    self.tokens -= 1
                    return True
            
            # Can't proceed
            if not blocking:
                return False
            
            # Check timeout
            if timeout is not None:
                if time.time() - start_time >= timeout:
                    return False
            
            # Wait a bit
            time.sleep(0.1)
    
    def get_wait_time(self) -> float:
        """Get estimated wait time until next token available."""
        with self.lock:
            if self.tokens >= 1:
                return 0.0
            
            tokens_needed = 1 - self.tokens
            return tokens_needed * self.window / self.rate


# Global rate limiter
_RATE_LIMITER = TokenBucketRateLimiter(RATE_LIMIT_RPM, RATE_LIMIT_WINDOW)


# ========================================================================================
# CIRCUIT BREAKER
# ========================================================================================

class CircuitBreaker:
    """Circuit breaker pattern for API calls."""
    
    def __init__(self, threshold: int, timeout: int):
        """
        Initialize circuit breaker.
        
        Args:
            threshold: Number of failures before opening
            timeout: Seconds before attempting to close
        """
        self.threshold = threshold
        self.timeout = timeout
        self.failures = 0
        self.last_failure_time = 0
        self.state = "closed"  # closed, open, half_open
        self.lock = threading.Lock()
    
    def call(self, func: Callable[[], Any]) -> Any:
        """
        Call function through circuit breaker.
        
        Args:
            func: Function to call
            
        Returns:
            Function result
            
        Raises:
            Exception: If circuit is open
        """
        with self.lock:
            # Check state
            if self.state == "open":
                # Check if timeout expired
                if time.time() - self.last_failure_time >= self.timeout:
                    LOGGER.info("Circuit breaker: transitioning to half-open")
                    self.state = "half_open"
                else:
                    raise Exception("Circuit breaker is OPEN - too many failures")
        
        # Try to call
        try:
            result = func()
            
            with self.lock:
                if self.state == "half_open":
                    LOGGER.info("Circuit breaker: transitioning to closed")
                    self.state = "closed"
                    self.failures = 0
            
            return result
            
        except Exception as e:
            with self.lock:
                self.failures += 1
                self.last_failure_time = time.time()
                
                if self.failures >= self.threshold:
                    LOGGER.warning(f"Circuit breaker: OPENING after {self.failures} failures")
                    self.state = "open"
            
            raise
    
    def reset(self) -> None:
        """Reset circuit breaker."""
        with self.lock:
            self.failures = 0
            self.state = "closed"
            LOGGER.info("Circuit breaker: RESET")


# Global circuit breaker
_CIRCUIT_BREAKER = CircuitBreaker(CIRCUIT_BREAKER_THRESHOLD, CIRCUIT_BREAKER_TIMEOUT)


# ========================================================================================
# CACHE
# ========================================================================================

class CompletionCache:
    """Cache for completions with TTL."""
    
    def __init__(self, ttl: int = CACHE_TTL):
        self.ttl = ttl
        self._cache: Dict[str, Tuple[str, float, CompletionMetrics]] = {}
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Tuple[str, CompletionMetrics]]:
        """Get from cache."""
        with self.lock:
            if key in self._cache:
                content, timestamp, metrics = self._cache[key]
                
                if time.time() - timestamp < self.ttl:
                    LOGGER.debug(f"Cache HIT: {key[:16]}...")
                    # Mark as cached
                    cached_metrics = CompletionMetrics(
                        prompt_tokens=metrics.prompt_tokens,
                        completion_tokens=metrics.completion_tokens,
                        total_tokens=metrics.total_tokens,
                        cost_usd=0.0,  # No cost for cached
                        duration_seconds=0.0,
                        model=metrics.model,
                        cached=True,
                        retries=0
                    )
                    return content, cached_metrics
                else:
                    del self._cache[key]
                    LOGGER.debug(f"Cache EXPIRED: {key[:16]}...")
        
        LOGGER.debug(f"Cache MISS: {key[:16]}...")
        return None
    
    def set(self, key: str, content: str, metrics: CompletionMetrics) -> None:
        """Set in cache."""
        with self.lock:
            self._cache[key] = (content, time.time(), metrics)
            LOGGER.debug(f"Cache SET: {key[:16]}...")
    
    def clear(self) -> None:
        """Clear cache."""
        with self.lock:
            self._cache.clear()
            LOGGER.info("Completion cache cleared")


# Global cache
_COMPLETION_CACHE = CompletionCache()


def compute_cache_key(
    system: str,
    user: str,
    model: str,
    temperature: float,
    max_tokens: int,
    response_format: str
) -> str:
    """
    Compute cache key for completion.
    
    Args:
        system: System prompt
        user: User prompt
        model: Model name
        temperature: Temperature
        max_tokens: Max tokens
        response_format: Response format
        
    Returns:
        SHA-256 hash
    """
    components = [
        system,
        user,
        model,
        str(temperature),
        str(max_tokens),
        response_format
    ]
    
    key_str = "|".join(components)
    return hashlib.sha256(key_str.encode()).hexdigest()


# ========================================================================================
# CLIENT MANAGEMENT
# ========================================================================================

_CLIENT_CACHE: Optional[OpenAI] = None
_CLIENT_LOCK = threading.Lock()


def get_secret(key: str) -> Optional[str]:
    """
    Get secret from Streamlit secrets or environment.
    
    Args:
        key: Secret key
        
    Returns:
        Secret value or None
    """
    # Try Streamlit secrets
    if HAS_STREAMLIT and st is not None:
        try:
            # Try nested openai.key
            if "openai" in st.secrets and key in st.secrets["openai"]:
                return str(st.secrets["openai"][key])
            
            # Try direct key
            if key in st.secrets:
                return str(st.secrets[key])
        except Exception:
            pass
    
    # Try environment
    return os.getenv(key)


def has_openai_key() -> bool:
    """
    Check if OpenAI API key is available.
    
    Returns:
        True if key available
    """
    return bool(get_secret("api_key") or get_secret("OPENAI_API_KEY"))


def build_client(config: OpenAIConfig) -> Optional[OpenAI]:
    """
    Build OpenAI client.
    
    Args:
        config: OpenAI configuration
        
    Returns:
        OpenAI client or None
    """
    if not config.api_key:
        return None
    
    kwargs: Dict[str, Any] = {"api_key": config.api_key}
    
    if config.base_url:
        kwargs["base_url"] = config.base_url
    
    if config.organization:
        kwargs["organization"] = config.organization
    
    if config.project:
        kwargs["project"] = config.project
    
    try:
        return OpenAI(**kwargs)  # type: ignore
    except Exception as e:
        LOGGER.error(f"Failed to build OpenAI client: {e}")
        return None


def get_client(config: Optional[OpenAIConfig] = None) -> Optional[OpenAI]:
    """
    Get singleton OpenAI client.
    
    Args:
        config: Optional configuration (uses secrets/env if None)
        
    Returns:
        OpenAI client or None
    """
    global _CLIENT_CACHE
    
    with _CLIENT_LOCK:
        if _CLIENT_CACHE is not None:
            return _CLIENT_CACHE
        
        # Build config if not provided
        if config is None:
            config = OpenAIConfig(
                api_key=get_secret("api_key") or get_secret("OPENAI_API_KEY"),
                base_url=get_secret("OPENAI_BASE_URL"),
                organization=get_secret("OPENAI_ORG"),
                project=get_secret("OPENAI_PROJECT")
            )
        
        client = build_client(config)
        _CLIENT_CACHE = client
        
        return client


def reset_client() -> None:
    """Reset cached client."""
    global _CLIENT_CACHE
    
    with _CLIENT_LOCK:
        _CLIENT_CACHE = None
        LOGGER.info("OpenAI client reset")


# ========================================================================================
# UTILITIES
# ========================================================================================

def choose_model(model: str) -> str:
    """
    Resolve model alias to actual model name.
    
    Args:
        model: Model name or alias
        
    Returns:
        Actual model name
    """
    return MODEL_ALIASES.get(model, model)


def estimate_cost(
    prompt_tokens: int,
    completion_tokens: int,
    model: str
) -> float:
    """
    Estimate cost in USD.
    
    Args:
        prompt_tokens: Input tokens
        completion_tokens: Output tokens
        model: Model name
        
    Returns:
        Estimated cost in USD
    """
    if model not in TOKEN_PRICING:
        LOGGER.warning(f"No pricing info for model {model}")
        return 0.0
    
    pricing = TOKEN_PRICING[model]
    
    input_cost = (prompt_tokens / 1_000_000) * pricing["input"]
    output_cost = (completion_tokens / 1_000_000) * pricing["output"]
    
    return input_cost + output_cost


def response_format_for(format_type: Optional[str]) -> Optional[Dict[str, str]]:
    """
    Get response format dict for API.
    
    Args:
        format_type: "json" or "text"
        
    Returns:
        Format dict or None
    """
    if format_type == "json":
        return {"type": "json_object"}
    
    return None


def is_transient_error(error: Exception) -> bool:
    """
    Check if error is transient (retryable).
    
    Args:
        error: Exception
        
    Returns:
        True if transient
    """
    return isinstance(error, (
        RateLimitError,
        APITimeoutError,
        APIConnectionError,
        APIError
    ))


def backoff_sleep(attempt: int, base: float = DEFAULT_BASE_BACKOFF) -> None:
    """
    Sleep with exponential backoff and jitter.
    
    Args:
        attempt: Attempt number (0-indexed)
        base: Base delay in seconds
    """
    wait = min(MAX_BACKOFF, base * (2 ** attempt))
    jitter = wait * (0.5 + random.random() * 0.5)
    
    LOGGER.debug(f"Backing off for {jitter:.2f}s (attempt {attempt + 1})")
    time.sleep(jitter)


def strip_json_noise(text: str) -> str:
    """
    Strip non-JSON prefix/suffix and extract first complete JSON object/array.
    
    Args:
        text: Text potentially containing JSON
        
    Returns:
        Extracted JSON string
    """
    if not isinstance(text, str):
        return text
    
    # Try to find JSON object or array
    match = re.search(r"(\{.*\}|\[.*\])", text, flags=re.DOTALL)
    
    if match:
        return match.group(1)
    
    return text


def safe_json_parse(text: str) -> Optional[dict]:
    """
    Safely parse JSON with fallback strategies.
    
    Args:
        text: JSON string
        
    Returns:
        Parsed dict or None
    """
    # Try direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    
    # Try after stripping noise
    try:
        cleaned = strip_json_noise(text)
        return json.loads(cleaned)
    except json.JSONDecodeError:
        pass
    
    LOGGER.warning("Failed to parse JSON")
    return None


# ========================================================================================
# MAIN API
# ========================================================================================

def chat_completion(
    system: str,
    user: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 1000,
    response_format: Literal["text", "json"] = "text",
    retries: int = DEFAULT_RETRIES,
    timeout: int = DEFAULT_TIMEOUT,
    *,
    stream: bool = False,
    on_token: Optional[Callable[[str], None]] = None,
    use_cache: bool = True,
    use_rate_limiter: bool = True,
    config: Optional[OpenAIConfig] = None
) -> str:
    """
    Call OpenAI Chat Completions API.
    
    Args:
        system: System prompt
        user: User prompt
        model: Model name
        temperature: Temperature (0-2)
        max_tokens: Max completion tokens
        response_format: "text" or "json"
        retries: Max retry attempts
        timeout: Timeout in seconds
        stream: Whether to stream response
        on_token: Callback for streaming tokens
        use_cache: Whether to use cache
        use_rate_limiter: Whether to use rate limiter
        config: Optional OpenAI config
        
    Returns:
        Completion text
    """
    result = chat_completion_with_metrics(
        system=system,
        user=user,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=response_format,
        retries=retries,
        timeout=timeout,
        stream=stream,
        on_token=on_token,
        use_cache=use_cache,
        use_rate_limiter=use_rate_limiter,
        config=config
    )
    
    return result.content


def chat_completion_with_metrics(
    system: str,
    user: str,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    max_tokens: int = 1000,
    response_format: Literal["text", "json"] = "text",
    retries: int = DEFAULT_RETRIES,
    timeout: int = DEFAULT_TIMEOUT,
    *,
    stream: bool = False,
    on_token: Optional[Callable[[str], None]] = None,
    use_cache: bool = True,
    use_rate_limiter: bool = True,
    config: Optional[OpenAIConfig] = None
) -> CompletionResult:
    """
    Call OpenAI API with full metrics tracking.
    
    Args:
        system: System prompt
        user: User prompt
        model: Model name
        temperature: Temperature
        max_tokens: Max tokens
        response_format: Response format
        retries: Max retries
        timeout: Timeout
        stream: Stream response
        on_token: Token callback
        use_cache: Use cache
        use_rate_limiter: Use rate limiter
        config: OpenAI config
        
    Returns:
        CompletionResult with content and metrics
    """
    start_time = time.time()
    
    # Get client
    client = get_client(config)
    if client is None:
        error_msg = "Brak klucza OpenAI. Ustaw OPENAI_API_KEY w secrets lub env."
        LOGGER.error(error_msg)
        
        # Return error result
        metrics = CompletionMetrics(
            prompt_tokens=0,
            completion_tokens=0,
            total_tokens=0,
            cost_usd=0.0,
            duration_seconds=time.time() - start_time,
            model=model,
            cached=False,
            retries=0
        )
        return CompletionResult(content=error_msg, metrics=metrics)
    
    # Resolve model
    actual_model = choose_model(model)
    
    # Check cache
    if use_cache and not stream:
        cache_key = compute_cache_key(
            system, user, actual_model, temperature, max_tokens, response_format
        )
        
        cached = _COMPLETION_CACHE.get(cache_key)
        if cached is not None:
            content, metrics = cached
            return CompletionResult(content=content, metrics=metrics)
    
    # Rate limiting
    if use_rate_limiter:
        wait_time = _RATE_LIMITER.get_wait_time()
        if wait_time > 0:
            LOGGER.debug(f"Rate limit: waiting {wait_time:.2f}s")
        
        acquired = _RATE_LIMITER.acquire(timeout=timeout)
        if not acquired:
            error_msg = "Rate limit exceeded - timeout waiting for token"
            LOGGER.warning(error_msg)
            
            metrics = CompletionMetrics(
                prompt_tokens=0,
                completion_tokens=0,
                total_tokens=0,
                cost_usd=0.0,
                duration_seconds=time.time() - start_time,
                model=actual_model,
                cached=False,
                retries=0
            )
            return CompletionResult(content=error_msg, metrics=metrics)
    
    # Response format
    resp_fmt = response_format_for(response_format)
    
    # Messages
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]
    
    # Retry loop with circuit breaker
    last_error: Optional[str] = None
    retry_count = 0
    
    for attempt in range(retries + 1):
        try:
            # Circuit breaker
            def api_call():
                if stream:
                    return _streaming_completion(
                        client=client,
                        model=actual_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        resp_fmt=resp_fmt,
                        timeout=timeout,
                        on_token=on_token
                    )
                else:
                    return _blocking_completion(
                        client=client,
                        model=actual_model,
                        messages=messages,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        resp_fmt=resp_fmt,
                        timeout=timeout
                    )
            
            response = _CIRCUIT_BREAKER.call(api_call)
            
            # Extract content and usage
            if isinstance(response, tuple):
                content, usage = response
            else:
                content = str(response)
                usage = None
            
            # Build metrics
            elapsed = time.time() - start_time
            
            prompt_tokens = usage.get("prompt_tokens", 0) if usage else 0
            completion_tokens = usage.get("completion_tokens", 0) if usage else 0
            total_tokens = usage.get("total_tokens", 0) if usage else 0
            
            cost = estimate_cost(prompt_tokens, completion_tokens, actual_model)
            
            metrics = CompletionMetrics(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost_usd=cost,
                duration_seconds=elapsed,
                model=actual_model,
                cached=False,
                retries=retry_count
            )
            
            # Cache result
            if use_cache and not stream:
                _COMPLETION_CACHE.set(cache_key, content, metrics)
            
            LOGGER.info(
                f"Completion success: {total_tokens} tokens, "
                f"${cost:.6f}, {elapsed:.2f}s"
            )
            
            return CompletionResult(content=content, metrics=metrics)
            
        except AuthenticationError as e:
            error_msg = "Błąd uwierzytelnienia OpenAI (sprawdź API key)"
            LOGGER.error(f"{error_msg}: {e}")
            last_error = error_msg
            break
            
        except BadRequestError as e:
            last_error = str(e)
            LOGGER.warning(f"Bad request: {e}")
            
            # Try without JSON format if that's the issue
            if "response_format" in last_error and response_format == "json":
                LOGGER.info("Retrying without JSON format enforcement")
                resp_fmt = None
                retry_count += 1
                continue
            
            break
            
        except Exception as e:
            last_error = str(e)
            LOGGER.warning(f"Attempt {attempt + 1} failed: {e}")
            
            if is_transient_error(e) and attempt < retries:
                retry_count += 1
                backoff_sleep(attempt)
                continue
            
            break
    
    # All retries failed
    error_msg = f"Błąd OpenAI po {retries + 1} próbach: {last_error or 'Nieznany błąd'}"
    LOGGER.error(error_msg)
    
    metrics = CompletionMetrics(
        prompt_tokens=0,
        completion_tokens=0,
        total_tokens=0,
        cost_usd=0.0,
        duration_seconds=time.time() - start_time,
        model=actual_model,
        cached=False,
        retries=retry_count
    )
    
    return CompletionResult(content=error_msg, metrics=metrics)


def _blocking_completion(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    resp_fmt: Optional[Dict[str, str]],
    timeout: int
) -> Tuple[str, Dict[str, int]]:
    """
    Blocking completion call.
    
    Args:
        client: OpenAI client
        model: Model name
        messages: Messages
        temperature: Temperature
        max_tokens: Max tokens
        resp_fmt: Response format
        timeout: Timeout
        
    Returns:
        Tuple (content, usage)
    """
    response = client.chat.completions.create(  # type: ignore
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=resp_fmt,
        timeout=timeout
    )
    
    content = response.choices[0].message.content  # type: ignore
    usage = {
        "prompt_tokens": response.usage.prompt_tokens,  # type: ignore
        "completion_tokens": response.usage.completion_tokens,  # type: ignore
        "total_tokens": response.usage.total_tokens  # type: ignore
    }
    
    if not isinstance(content, str):
        content = json.dumps(content, ensure_ascii=False)
    
    return content.strip(), usage


def _streaming_completion(
    client: OpenAI,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float,
    max_tokens: int,
    resp_fmt: Optional[Dict[str, str]],
    timeout: int,
    on_token: Optional[Callable[[str], None]]
) -> Tuple[str, Dict[str, int]]:
    """
    Streaming completion call.
    
    Args:
        client: OpenAI client
        model: Model name
        messages: Messages
        temperature: Temperature
        max_tokens: Max tokens
        resp_fmt: Response format
        timeout: Timeout
        on_token: Token callback
        
    Returns:
        Tuple (content, usage)
    """
    accumulated: List[str] = []
    usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    
    stream = client.chat.completions.create(  # type: ignore
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        response_format=resp_fmt,
        timeout=timeout,
        stream=True
    )
    
    for chunk in stream:
        # Extract content
        if hasattr(chunk.choices[0].delta, "content"):
            token = chunk.choices[0].delta.content
            
            if token:
                accumulated.append(token)
                
                if on_token:
                    try:
                        on_token(token)
                    except Exception as e:
                        LOGGER.warning(f"Token callback failed: {e}")
        
        # Extract usage (usually in last chunk)
        if hasattr(chunk, "usage") and chunk.usage:
            usage = {
                "prompt_tokens": chunk.usage.prompt_tokens,
                "completion_tokens": chunk.usage.completion_tokens,
                "total_tokens": chunk.usage.total_tokens
            }
    
    content = "".join(accumulated).strip()
    
    return content, usage


def chat_completion_json(
    system: str,
    user: str,
    **kwargs: Any
) -> Dict[str, Any]:
    """
    Call API with JSON response format.
    
    Args:
        system: System prompt
        user: User prompt
        **kwargs: Additional arguments
        
    Returns:
        Parsed JSON dict
    """
    text = chat_completion(
        system=system,
        user=user,
        response_format="json",
        **kwargs
    )
    
    parsed = safe_json_parse(text)
    
    if isinstance(parsed, dict):
        return parsed
    
    return {
        "error": "Nie udało się sparsować JSON",
        "raw_text": text
    }


# ========================================================================================
# UTILITIES & MANAGEMENT
# ========================================================================================

def clear_completion_cache() -> None:
    """Clear completion cache."""
    _COMPLETION_CACHE.clear()


def reset_circuit_breaker() -> None:
    """Reset circuit breaker."""
    _CIRCUIT_BREAKER.reset()


def get_cache_stats() -> Dict[str, Any]:
    """
    Get cache statistics.
    
    Returns:
        Stats dict
    """
    return {
        "size": len(_COMPLETION_CACHE._cache),
        "ttl": _COMPLETION_CACHE.ttl
    }


def get_rate_limiter_stats() -> Dict[str, Any]:
    """
    Get rate limiter statistics.
    
    Returns:
        Stats dict
    """
    return {
        "tokens": _RATE_LIMITER.tokens,
        "rate": _RATE_LIMITER.rate,
        "window": _RATE_LIMITER.window,
        "wait_time": _RATE_LIMITER.get_wait_time()
    }


def get_circuit_breaker_stats() -> Dict[str, Any]:
    """
    Get circuit breaker statistics.
    
    Returns:
        Stats dict
    """
    return {
        "state": _CIRCUIT_BREAKER.state,
        "failures": _CIRCUIT_BREAKER.failures,
        "threshold": _CIRCUIT_BREAKER.threshold
    }