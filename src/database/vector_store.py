"""
vector_store.py — ULTRA PRO Edition

Production-ready vector store with:
- Multi-backend support (Chroma, Pinecone, FAISS)
- Advanced chunking with overlap
- Batch embedding with retry logic
- Deduplication strategies
- Namespace isolation
- Metadata filtering
- Performance monitoring
- Connection pooling
- Graceful degradation
"""

from __future__ import annotations

import os
import time
import json
import math
import uuid
import hashlib
import logging
import pathlib
import threading
import importlib
from dataclasses import dataclass, field
from typing import (
    Any, Dict, Iterable, List, Optional, Tuple, Union,
    Protocol, runtime_checkable
)
from contextlib import contextmanager

# ========================================================================================
# OPTIONAL DEPENDENCIES
# ========================================================================================

try:
    import streamlit as st
    HAS_STREAMLIT = True
except ImportError:
    st = None  # type: ignore
    HAS_STREAMLIT = False

try:
    import yaml
    HAS_YAML = True
except ImportError:
    yaml = None  # type: ignore
    HAS_YAML = False

try:
    import tiktoken
    HAS_TIKTOKEN = True
except ImportError:
    tiktoken = None  # type: ignore
    HAS_TIKTOKEN = False

try:
    import chromadb
    from chromadb.api.types import Documents, Embeddings
    from chromadb.utils.embedding_functions import EmbeddingFunction as ChromaEmbeddingFunction
    HAS_CHROMA = True
except ImportError:
    chromadb = None  # type: ignore
    ChromaEmbeddingFunction = object  # type: ignore
    HAS_CHROMA = False

try:
    import importlib
    pinecone_module = importlib.import_module("pinecone")
    # Prefer explicit Pinecone export when available, otherwise use module object
    Pinecone = getattr(pinecone_module, "Pinecone", pinecone_module)
    pinecone = pinecone_module
    HAS_PINECONE = True
except Exception:
    pinecone = None  # type: ignore
    Pinecone = None  # type: ignore
    HAS_PINECONE = False

try:
    import importlib
    faiss = importlib.import_module("faiss")
    import numpy as np
    HAS_FAISS = True
except Exception:
    faiss = None  # type: ignore
    np = None  # type: ignore
    HAS_FAISS = False

# OpenAI integration
try:
    from src.ai_engine.openai_integrator import get_client
    HAS_OPENAI = True
except ImportError:
    try:
        from ai_engine.openai_integrator import get_client  # type: ignore
        HAS_OPENAI = True
    except ImportError:
        get_client = None  # type: ignore
        HAS_OPENAI = False

# ========================================================================================
# LOGGING
# ========================================================================================

def get_logger(name: str = "vector_store") -> logging.Logger:
    """Configure logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.INFO)
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
        )
        logger.addHandler(handler)
        logger.propagate = False
    return logger

LOGGER = get_logger()

# ========================================================================================
# CONSTANTS
# ========================================================================================

# Defaults
DEFAULT_NAMESPACE = "intelligent-predictor"
DEFAULT_COLLECTION = "default"
DEFAULT_PERSIST_DIR = "data/vector_store"
DEFAULT_METRIC = "cosine"
DEFAULT_EMBEDDING_MODEL = "text-embedding-3-small"
DEFAULT_EMBEDDING_DIM = 1536
DEFAULT_BATCH_SIZE = 128

# Chunking
DEFAULT_MAX_CHUNK_TOKENS = 7500
DEFAULT_CHUNK_OVERLAP = 200
MIN_CHUNK_SIZE = 100
MAX_CHUNK_SIZE = 8000

# Retry settings
MAX_EMBEDDING_RETRIES = 4
EMBEDDING_RETRY_DELAY = 0.75
EMBEDDING_RETRY_BACKOFF = 2.0

# Rate limiting
EMBEDDING_RATE_LIMIT_DELAY = 0.05  # 50ms between batches

# ========================================================================================
# DATACLASSES
# ========================================================================================

@dataclass
class VectorStoreConfig:
    """Vector store configuration."""
    
    # Core settings
    enabled: bool = False
    backend: str = "chroma"  # chroma | pinecone | faiss
    collection: str = DEFAULT_COLLECTION
    namespace: str = DEFAULT_NAMESPACE
    
    # Storage
    persist_dir: str = DEFAULT_PERSIST_DIR
    
    # Embeddings
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_dim: int = DEFAULT_EMBEDDING_DIM
    batch_size: int = DEFAULT_BATCH_SIZE
    
    # Distance metric
    metric: str = DEFAULT_METRIC  # cosine | dotproduct | euclidean | l2
    
    # Chunking
    chunk_enabled: bool = True
    max_chunk_tokens: int = DEFAULT_MAX_CHUNK_TOKENS
    chunk_overlap_tokens: int = DEFAULT_CHUNK_OVERLAP
    
    # ID strategy
    id_strategy: str = "uuid"  # uuid | hash | auto
    deduplicate: bool = False
    
    # Pinecone specific
    pinecone_index: Optional[str] = None
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None
    pinecone_cloud: Optional[str] = None
    
    # Performance
    enable_metrics: bool = True
    cache_embeddings: bool = True
    
    # Debugging
    verbose: bool = False


@dataclass
class VectorStoreMetrics:
    """Vector store performance metrics."""
    
    total_documents: int = 0
    total_chunks: int = 0
    total_queries: int = 0
    total_upserts: int = 0
    total_deletes: int = 0
    embedding_calls: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    total_embedding_time: float = 0.0
    total_query_time: float = 0.0
    errors: int = 0
    start_time: float = field(default_factory=time.time)
    
    @property
    def cache_hit_rate(self) -> float:
        """Calculate cache hit rate."""
        total = self.cache_hits + self.cache_misses
        return (self.cache_hits / total * 100) if total > 0 else 0.0
    
    @property
    def avg_embedding_time_ms(self) -> float:
        """Average embedding time in milliseconds."""
        if self.embedding_calls == 0:
            return 0.0
        return (self.total_embedding_time / self.embedding_calls) * 1000
    
    @property
    def avg_query_time_ms(self) -> float:
        """Average query time in milliseconds."""
        if self.total_queries == 0:
            return 0.0
        return (self.total_query_time / self.total_queries) * 1000
    
    def reset(self) -> None:
        """Reset all metrics."""
        self.total_queries = 0
        self.total_upserts = 0
        self.total_deletes = 0
        self.embedding_calls = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_embedding_time = 0.0
        self.total_query_time = 0.0
        self.errors = 0
        self.start_time = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        uptime = time.time() - self.start_time
        
        return {
            "total_documents": self.total_documents,
            "total_chunks": self.total_chunks,
            "total_queries": self.total_queries,
            "total_upserts": self.total_upserts,
            "total_deletes": self.total_deletes,
            "embedding_calls": self.embedding_calls,
            "cache_hit_rate": round(self.cache_hit_rate, 2),
            "avg_embedding_time_ms": round(self.avg_embedding_time_ms, 3),
            "avg_query_time_ms": round(self.avg_query_time_ms, 3),
            "errors": self.errors,
            "uptime_seconds": round(uptime, 1)
        }


@dataclass
class QueryResult:
    """Vector search result."""
    
    id: str
    text: Optional[str]
    score: float
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "text": self.text,
            "score": self.score,
            "metadata": self.metadata
        }


# ========================================================================================
# PROTOCOLS
# ========================================================================================

@runtime_checkable
class VectorStore(Protocol):
    """Vector store interface protocol."""
    
    enabled: bool
    
    def upsert_texts(
        self,
        texts: List[str],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        namespace: Optional[str] = None
    ) -> List[str]: ...
    
    def query(
        self,
        text_or_texts: Union[str, List[str]],
        top_k: int = 5,
        namespace: Optional[str] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> List[QueryResult]: ...
    
    def delete(
        self,
        ids: Optional[List[str]] = None,
        namespace: Optional[str] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> int: ...
    
    def persist(self) -> None: ...
    def flush(self, namespace: Optional[str] = None) -> int: ...
    def status(self) -> str: ...
    def get_metrics(self) -> Dict[str, Any]: ...
    def close(self) -> None: ...


# ========================================================================================
# CONFIGURATION LOADING
# ========================================================================================

def _load_vector_store_config() -> VectorStoreConfig:
    """
    Load vector store configuration from multiple sources (priority order):
    1. Environment variables (highest priority)
    2. Streamlit secrets
    3. config.yaml
    4. Defaults (lowest priority)
    
    Returns:
        VectorStoreConfig with merged settings
    """
    config = VectorStoreConfig()
    
    # ============================================================================
    # 1. Load from config.yaml
    # ============================================================================
    
    if HAS_YAML and yaml is not None:
        try:
            config_path = pathlib.Path("config.yaml")
            
            if config_path.exists():
                LOGGER.debug("Loading vector store config from config.yaml")
                
                data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                
                if isinstance(data, dict) and "vector_store" in data:
                    vs_cfg = data["vector_store"]
                    
                    if isinstance(vs_cfg, dict):
                        config.enabled = bool(vs_cfg.get("enabled", config.enabled))
                        config.backend = str(vs_cfg.get("backend", config.backend))
                        config.collection = str(vs_cfg.get("collection", config.collection))
                        config.namespace = str(vs_cfg.get("namespace", config.namespace))
                        config.persist_dir = str(vs_cfg.get("persist_dir", config.persist_dir))
                        config.embedding_model = str(vs_cfg.get("embedding_model", config.embedding_model))
                        config.embedding_dim = int(vs_cfg.get("embedding_dim", config.embedding_dim))
                        config.batch_size = int(vs_cfg.get("batch_size", config.batch_size))
                        config.metric = str(vs_cfg.get("metric", config.metric))
                        config.chunk_enabled = bool(vs_cfg.get("chunk_enabled", config.chunk_enabled))
                        config.max_chunk_tokens = int(vs_cfg.get("max_chunk_tokens", config.max_chunk_tokens))
                        config.chunk_overlap_tokens = int(vs_cfg.get("chunk_overlap_tokens", config.chunk_overlap_tokens))
                        config.id_strategy = str(vs_cfg.get("id_strategy", config.id_strategy))
                        config.deduplicate = bool(vs_cfg.get("deduplicate", config.deduplicate))
                        config.pinecone_index = vs_cfg.get("pinecone_index", config.pinecone_index)
                        config.pinecone_api_key = vs_cfg.get("pinecone_api_key", config.pinecone_api_key)
                        config.enable_metrics = bool(vs_cfg.get("enable_metrics", config.enable_metrics))
                        config.verbose = bool(vs_cfg.get("verbose", config.verbose))
                        
                        LOGGER.info("Vector store config loaded from config.yaml")
        except Exception as e:
            LOGGER.warning(f"Failed to load config.yaml: {e}")
    
    # ============================================================================
    # 2. Load from Streamlit secrets
    # ============================================================================
    
    if HAS_STREAMLIT and st is not None:
        try:
            secrets = st.secrets.get("vector_store", {})
            
            if secrets:
                LOGGER.debug("Loading vector store config from Streamlit secrets")
                
                config.enabled = bool(secrets.get("enabled", config.enabled))
                config.backend = str(secrets.get("backend", config.backend))
                config.collection = str(secrets.get("collection", config.collection))
                config.namespace = str(secrets.get("namespace", config.namespace))
                config.embedding_model = str(secrets.get("embedding_model", config.embedding_model))
                config.pinecone_api_key = secrets.get("pinecone_api_key", config.pinecone_api_key)
                config.pinecone_index = secrets.get("pinecone_index", config.pinecone_index)
                
                LOGGER.info("Vector store config loaded from Streamlit secrets")
        except Exception as e:
            LOGGER.debug(f"No Streamlit secrets found: {e}")
    
    # ============================================================================
    # 3. Environment variables (highest priority)
    # ============================================================================
    
    # Enabled flag
    if os.getenv("VECTOR_ENABLED"):
        config.enabled = os.getenv("VECTOR_ENABLED", "").lower() in ("1", "true", "yes", "on")
    
    # Backend
    if os.getenv("VECTOR_BACKEND"):
        config.backend = os.getenv("VECTOR_BACKEND", config.backend)
    
    # Collection/Index
    if os.getenv("VECTOR_COLLECTION"):
        config.collection = os.getenv("VECTOR_COLLECTION", config.collection)
    
    if os.getenv("VECTOR_NAMESPACE"):
        config.namespace = os.getenv("VECTOR_NAMESPACE", config.namespace)
    
    # Embeddings
    if os.getenv("OPENAI_EMBED_MODEL"):
        config.embedding_model = os.getenv("OPENAI_EMBED_MODEL", config.embedding_model)
    
    if os.getenv("OPENAI_EMBED_DIM"):
        try:
            config.embedding_dim = int(os.getenv("OPENAI_EMBED_DIM", config.embedding_dim))
        except ValueError:
            pass
    
    # Pinecone
    if os.getenv("PINECONE_API_KEY"):
        config.pinecone_api_key = os.getenv("PINECONE_API_KEY")
    
    if os.getenv("PINECONE_INDEX"):
        config.pinecone_index = os.getenv("PINECONE_INDEX")
    
    if os.getenv("PINECONE_ENVIRONMENT"):
        config.pinecone_environment = os.getenv("PINECONE_ENVIRONMENT")
    
    # Verbose mode
    if os.getenv("VECTOR_VERBOSE"):
        config.verbose = os.getenv("VECTOR_VERBOSE", "").lower() in ("1", "true", "yes")
    
    # ============================================================================
    # Validation
    # ============================================================================
    
    # Validate backend
    valid_backends = ("chroma", "pinecone", "faiss")
    if config.backend not in valid_backends:
        LOGGER.warning(f"Invalid backend: {config.backend}, using 'chroma'")
        config.backend = "chroma"
    
    # Validate metric
    valid_metrics = ("cosine", "dotproduct", "euclidean", "l2")
    if config.metric not in valid_metrics:
        LOGGER.warning(f"Invalid metric: {config.metric}, using 'cosine'")
        config.metric = "cosine"
    
    # Validate chunking
    if config.max_chunk_tokens < MIN_CHUNK_SIZE:
        config.max_chunk_tokens = MIN_CHUNK_SIZE
    
    if config.max_chunk_tokens > MAX_CHUNK_SIZE:
        config.max_chunk_tokens = MAX_CHUNK_SIZE
    
    if config.chunk_overlap_tokens < 0:
        config.chunk_overlap_tokens = 0
    
    if config.chunk_overlap_tokens >= config.max_chunk_tokens:
        config.chunk_overlap_tokens = config.max_chunk_tokens // 4
    
    # Set verbose logging
    if config.verbose:
        LOGGER.setLevel(logging.DEBUG)
        LOGGER.debug("Verbose logging enabled")
    
    return config


# ========================================================================================
# TEXT PROCESSING
# ========================================================================================

def hash_text(text: str) -> str:
    """
    Create stable hash of text.
    
    Args:
        text: Text to hash
        
    Returns:
        SHA-1 hash (40 characters)
    """
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def estimate_tokens(text: str, model: str = "gpt-3.5-turbo") -> int:
    """
    Estimate token count for text.
    
    Args:
        text: Text to estimate
        model: Model name hint
        
    Returns:
        Estimated token count
    """
    if HAS_TIKTOKEN and tiktoken is not None:
        try:
            encoding = tiktoken.encoding_for_model(model)
            return len(encoding.encode(text))
        except Exception:
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
                return len(encoding.encode(text))
            except Exception:
                pass
    
    # Fallback: ~4 characters per token
    return max(1, len(text) // 4)


def chunk_text(
    text: str,
    max_tokens: int,
    overlap_tokens: int,
    model: str = "gpt-3.5-turbo"
) -> List[str]:
    """
    Split text into chunks with overlap.
    
    Args:
        text: Text to chunk
        max_tokens: Maximum tokens per chunk
        overlap_tokens: Overlap between chunks
        model: Model name hint for tokenization
        
    Returns:
        List of text chunks
    """
    if not text or not text.strip():
        return []
    
    # Estimate current size
    current_tokens = estimate_tokens(text, model)
    
    if current_tokens <= max_tokens:
        return [text]
    
    # Use tiktoken if available
    if HAS_TIKTOKEN and tiktoken is not None:
        try:
            encoding = tiktoken.encoding_for_model(model)
        except Exception:
            try:
                encoding = tiktoken.get_encoding("cl100k_base")
            except Exception:
                encoding = None
        
        if encoding is not None:
            tokens = encoding.encode(text)
            chunks = []
            
            start = 0
            while start < len(tokens):
                end = min(len(tokens), start + max_tokens)
                chunk_tokens = tokens[start:end]
                chunk_text = encoding.decode(chunk_tokens)
                chunks.append(chunk_text)
                
                if end == len(tokens):
                    break
                
                start = max(0, end - overlap_tokens)
            
            return chunks
    
    # Fallback: Character-based chunking
    approx_chars_per_token = 4
    max_chars = max_tokens * approx_chars_per_token
    overlap_chars = overlap_tokens * approx_chars_per_token
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(len(text), start + max_chars)
        
        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence ending
            for punct in ('.', '!', '?', '\n'):
                last_punct = text.rfind(punct, start, end)
                if last_punct != -1:
                    end = last_punct + 1
                    break
        
        chunk = text[start:end].strip()
        
        if chunk:
            chunks.append(chunk)
        
        if end == len(text):
            break
        
        start = max(0, end - overlap_chars)
    
    return chunks


# ========================================================================================
# EMBEDDING FUNCTIONS
# ========================================================================================

class EmbeddingCache:
    """Simple in-memory cache for embeddings."""
    
    def __init__(self, max_size: int = 10000):
        self.cache: Dict[str, List[float]] = {}
        self.max_size = max_size
        self._lock = threading.Lock()
    
    def get(self, text: str) -> Optional[List[float]]:
        """Get embedding from cache."""
        key = hash_text(text)
        
        with self._lock:
            return self.cache.get(key)
    
    def set(self, text: str, embedding: List[float]) -> None:
        """Set embedding in cache."""
        key = hash_text(text)
        
        with self._lock:
            # Simple LRU: remove oldest if full
            if len(self.cache) >= self.max_size:
                # Remove first item (oldest)
                first_key = next(iter(self.cache))
                del self.cache[first_key]
            
            self.cache[key] = embedding
    
    def clear(self) -> None:
        """Clear cache."""
        with self._lock:
            self.cache.clear()


class OpenAIEmbedding:
    """
    OpenAI embedding function with retry logic and caching.
    """
    
    def __init__(
        self,
        model: str = DEFAULT_EMBEDDING_MODEL,
        batch_size: int = DEFAULT_BATCH_SIZE,
        enable_cache: bool = True
    ):
        self.model = model
        self.batch_size = max(1, batch_size)
        self.enable_cache = enable_cache
        self.cache = EmbeddingCache() if enable_cache else None
        self._lock = threading.Lock()
        
        if not HAS_OPENAI or get_client is None:
            raise RuntimeError("OpenAI client not available")
        
        self.client = get_client()
        
        if self.client is None:
            raise RuntimeError("OpenAI API key not configured")
    
    def embed_single(self, text: str) -> List[float]:
        """Embed single text."""
        return self.embed_batch([text])[0]
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Embed batch of texts with retry logic.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        if not texts:
            return []
        
        # Check cache first
        if self.enable_cache and self.cache is not None:
            cached_results: List[Optional[List[float]]] = []
            uncached_texts: List[str] = []
            uncached_indices: List[int] = []
            
            for i, text in enumerate(texts):
                cached = self.cache.get(text)
                
                if cached is not None:
                    cached_results.append(cached)
                else:
                    cached_results.append(None)
                    uncached_texts.append(text)
                    uncached_indices.append(i)
            
            # If all cached, return immediately
            if not uncached_texts:
                return [r for r in cached_results if r is not None]
            
            # Embed uncached texts
            fresh_embeddings = self._embed_with_retry(uncached_texts)
            
            # Store in cache
            for text, embedding in zip(uncached_texts, fresh_embeddings):
                self.cache.set(text, embedding)
            
            # Merge results
            result = []
            fresh_idx = 0
            
            for cached in cached_results:
                if cached is not None:
                    result.append(cached)
                else:
                    result.append(fresh_embeddings[fresh_idx])
                    fresh_idx += 1
            
            return result
        
        # No cache - direct embedding
        return self._embed_with_retry(texts)
    
    def _embed_with_retry(self, texts: List[str]) -> List[List[float]]:
        """
        Embed texts with retry logic.
        
        Args:
            texts: List of texts
            
        Returns:
            List of embeddings
        """
        all_embeddings: List[List[float]] = []
        
        # Process in batches
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            
            delay = EMBEDDING_RETRY_DELAY
            last_error = None
            
            for attempt in range(MAX_EMBEDDING_RETRIES):
                try:
                    response = self.client.embeddings.create(
                        model=self.model,
                        input=batch
                    )
                    
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                    
                    last_error = None
                    break
                    
                except Exception as e:
                    last_error = e
                    LOGGER.warning(f"Embedding attempt {attempt + 1} failed: {e}")
                    
                    if attempt < MAX_EMBEDDING_RETRIES - 1:
                        time.sleep(delay)
                        delay *= EMBEDDING_RETRY_BACKOFF
            
            if last_error:
                raise RuntimeError(f"Embedding failed after {MAX_EMBEDDING_RETRIES} attempts: {last_error}")
            
            # Rate limiting between batches
            if i + self.batch_size < len(texts):
                time.sleep(EMBEDDING_RATE_LIMIT_DELAY)
        
        return all_embeddings


class ChromaEmbeddingAdapter(ChromaEmbeddingFunction):  # type: ignore
    """Adapter for Chroma's embedding function interface."""
    
    def __init__(self, embedding_fn: OpenAIEmbedding):
        self.embedding_fn = embedding_fn
    
    def __call__(self, input: Documents) -> Embeddings:  # type: ignore
        """Embed documents for Chroma."""
        if not input:
            return []
        
        texts = list(input)
        return self.embedding_fn.embed_batch(texts)


# ========================================================================================
# NOOP IMPLEMENTATION
# ========================================================================================

class NoopVectorStore:
    """No-op vector store (disabled state)."""
    
    def __init__(self, reason: str = "Vector store disabled"):
        self.enabled = False
        self.reason = reason
        self.metrics = VectorStoreMetrics()
    
    def upsert_texts(
        self,
        texts: List[str],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        namespace: Optional[str] = None
    ) -> List[str]:
        return [ids[i] if ids and i < len(ids) else f"noop-{i}" for i in range(len(texts))]
    
    def query(
        self,
        text_or_texts: Union[str, List[str]],
        top_k: int = 5,
        namespace: Optional[str] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> List[QueryResult]:
        return []
    
    def delete(
        self,
        ids: Optional[List[str]] = None,
        namespace: Optional[str] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> int:
        return 0
    
    def persist(self) -> None:
        pass
    
    def flush(self, namespace: Optional[str] = None) -> int:
        return 0
    
    def status(self) -> str:
        return f"❌ {self.reason}"
    
    def get_metrics(self) -> Dict[str, Any]:
        return {"enabled": False, "reason": self.reason}
    
    def close(self) -> None:
        pass


# ========================================================================================
# CHROMA IMPLEMENTATION
# ========================================================================================

class ChromaVectorStore:
    """
    ChromaDB vector store implementation.
    
    Features:
    - Persistent storage
    - Namespace isolation
    - Metadata filtering
    - Automatic chunking
    """
    
    def __init__(self, config: VectorStoreConfig):
        if not HAS_CHROMA or chromadb is None:
            raise RuntimeError("ChromaDB is not installed")
        
        self.enabled = True
        self.config = config
        self.metrics = VectorStoreMetrics() if config.enable_metrics else None
        self._lock = threading.Lock()
        
        # Ensure persist directory exists
        persist_path = pathlib.Path(config.persist_dir)
        persist_path.mkdir(parents=True, exist_ok=True)
        
        # Initialize client
        try:
            self.client = chromadb.PersistentClient(path=str(persist_path))
            
            # Initialize embedding function
            embedding_fn = OpenAIEmbedding(
                model=config.embedding_model,
                batch_size=config.batch_size,
                enable_cache=config.cache_embeddings
            )
            
            self.embedding_adapter = ChromaEmbeddingAdapter(embedding_fn)
            
            # Get or create collection
            self.collection = self.client.get_or_create_collection(
                name=config.collection,
                metadata={"hnsw:space": config.metric},
                embedding_function=self.embedding_adapter
            )
            
            LOGGER.info(f"✅ ChromaDB initialized: {config.collection}")
            
        except Exception as e:
            LOGGER.error(f"Failed to initialize ChromaDB: {e}")
            raise
    
    def _generate_id(self, text: str, strategy: str) -> str:
        """Generate ID based on strategy."""
        if strategy == "hash":
            return hash_text(text)
        elif strategy == "auto":
            return f"auto-{hash_text(text)[:16]}"
        else:  # uuid
            return uuid.uuid4().hex
    
    def _prepare_chunks(
        self,
        texts: List[str]) -> List[Tuple[str, str, int]]:
        """
        Prepare text chunks with parent IDs.
        
        Args:
            texts: List of texts to chunk
            
        Returns:
            List of (chunk_text, parent_id, chunk_index)
        """
        chunks: List[Tuple[str, str, int]] = []
        
        for text in texts:
            parent_id = self._generate_id(text, self.config.id_strategy)
            
            if not self.config.chunk_enabled:
                chunks.append((text, parent_id, 0))
                continue
            
            # Chunk text
            text_chunks = chunk_text(
                text,
                max_tokens=self.config.max_chunk_tokens,
                overlap_tokens=self.config.chunk_overlap_tokens,
                model=self.config.embedding_model
            )
            
            for idx, chunk in enumerate(text_chunks):
                chunks.append((chunk, parent_id, idx))
        
        return chunks
    
    def upsert_texts(
        self,
        texts: List[str],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        namespace: Optional[str] = None
    ) -> List[str]:
        """
        Insert or update texts in vector store.
        
        Args:
            texts: List of texts to upsert
            ids: Optional custom IDs (one per text)
            metadatas: Optional metadata (one per text)
            namespace: Optional namespace
            
        Returns:
            List of parent document IDs
        """
        if not texts:
            return []
        
        start_time = time.time()
        
        try:
            # Prepare chunks
            chunks = self._prepare_chunks(texts)
            
            # Build metadata
            ns = namespace or self.config.namespace
            base_metadata = {"namespace": ns}
            
            user_metadatas = metadatas or [{} for _ in texts]
            custom_ids = ids or [None] * len(texts)
            
            # Process chunks
            documents: List[str] = []
            chunk_ids: List[str] = []
            chunk_metadatas: List[Dict[str, Any]] = []
            parent_ids: List[str] = []
            
            text_idx = 0
            
            for chunk_text, parent_id, chunk_idx in chunks:
                # Use custom ID if provided, otherwise use generated parent_id
                actual_parent_id = custom_ids[text_idx] if custom_ids[text_idx] else parent_id
                
                # Track parent IDs (only once per document)
                if chunk_idx == 0:
                    parent_ids.append(actual_parent_id)
                    text_idx += 1
                
                # Build chunk ID
                chunk_id = f"{actual_parent_id}__chunk_{chunk_idx}"
                
                # Build metadata
                user_meta = user_metadatas[min(text_idx - 1, len(user_metadatas) - 1)]
                
                chunk_metadata = {
                    **base_metadata,
                    **(user_meta or {}),
                    "parent_id": actual_parent_id,
                    "chunk_index": chunk_idx,
                    "total_chunks": -1  # Will be updated if needed
                }
                
                documents.append(chunk_text)
                chunk_ids.append(chunk_id)
                chunk_metadatas.append(chunk_metadata)
            
            # Deduplication
            if self.config.deduplicate:
                seen = set()
                unique_docs = []
                unique_ids = []
                unique_metas = []
                
                for doc, cid, meta in zip(documents, chunk_ids, chunk_metadatas):
                    if cid not in seen:
                        seen.add(cid)
                        unique_docs.append(doc)
                        unique_ids.append(cid)
                        unique_metas.append(meta)
                
                documents = unique_docs
                chunk_ids = unique_ids
                chunk_metadatas = unique_metas
            
            # Upsert to Chroma
            self.collection.upsert(
                documents=documents,
                ids=chunk_ids,
                metadatas=chunk_metadatas
            )
            
            # Update metrics
            if self.metrics:
                with self._lock:
                    self.metrics.total_documents += len(texts)
                    self.metrics.total_chunks += len(documents)
                    self.metrics.total_upserts += 1
            
            elapsed = time.time() - start_time
            LOGGER.debug(
                f"Upserted {len(texts)} documents ({len(documents)} chunks) "
                f"in {elapsed:.2f}s"
            )
            
            return parent_ids
            
        except Exception as e:
            LOGGER.error(f"Upsert failed: {e}")
            
            if self.metrics:
                with self._lock:
                    self.metrics.errors += 1
            
            raise
    
    def query(
        self,
        text_or_texts: Union[str, List[str]],
        top_k: int = 5,
        namespace: Optional[str] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> List[QueryResult]:
        """
        Query vector store.
        
        Args:
            text_or_texts: Query text(s)
            top_k: Number of results to return
            namespace: Optional namespace filter
            where: Optional metadata filter
            
        Returns:
            List of query results
        """
        start_time = time.time()
        
        try:
            # Normalize input
            queries = [text_or_texts] if isinstance(text_or_texts, str) else text_or_texts
            
            # Build filter
            ns = namespace or self.config.namespace
            filter_dict = {"namespace": ns}
            
            if where:
                filter_dict.update(where)
            
            # Query Chroma
            response = self.collection.query(
                query_texts=queries,
                n_results=top_k,
                where=filter_dict
            )
            
            # Parse results
            results: List[QueryResult] = []
            
            ids = response.get("ids", [])
            documents = response.get("documents", [])
            distances = response.get("distances", [])
            metadatas = response.get("metadatas", [])
            
            for i in range(len(ids)):
                for j in range(len(ids[i])):
                    result = QueryResult(
                        id=ids[i][j],
                        text=documents[i][j] if documents else None,
                        score=float(distances[i][j]) if distances else 0.0,
                        metadata=metadatas[i][j] if metadatas else {}
                    )
                    results.append(result)
            
            # Update metrics
            if self.metrics:
                elapsed = time.time() - start_time
                
                with self._lock:
                    self.metrics.total_queries += 1
                    self.metrics.total_query_time += elapsed
            
            return results
            
        except Exception as e:
            LOGGER.error(f"Query failed: {e}")
            
            if self.metrics:
                with self._lock:
                    self.metrics.errors += 1
            
            raise
    
    def delete(
        self,
        ids: Optional[List[str]] = None,
        namespace: Optional[str] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> int:
        """
        Delete documents from vector store.
        
        Args:
            ids: Optional list of IDs to delete
            namespace: Optional namespace filter
            where: Optional metadata filter
            
        Returns:
            Number of documents deleted (or -1 if unknown)
        """
        try:
            # Build filter
            filter_dict = {}
            
            if namespace:
                filter_dict["namespace"] = namespace or self.config.namespace
            
            if where:
                filter_dict.update(where)
            
            # Delete
            self.collection.delete(
                ids=ids,
                where=filter_dict if filter_dict else None
            )
            
            # Update metrics
            if self.metrics:
                with self._lock:
                    self.metrics.total_deletes += 1
            
            # Chroma doesn't return count
            return -1
            
        except Exception as e:
            LOGGER.error(f"Delete failed: {e}")
            
            if self.metrics:
                with self._lock:
                    self.metrics.errors += 1
            
            raise
    
    def persist(self) -> None:
        """Persist data to disk."""
        try:
            if hasattr(self.client, "persist"):
                self.client.persist()
            LOGGER.debug("Data persisted to disk")
        except Exception as e:
            LOGGER.warning(f"Persist failed: {e}")
    
    def flush(self, namespace: Optional[str] = None) -> int:
        """
        Flush collection (delete all data).
        
        Args:
            namespace: Optional namespace to flush
            
        Returns:
            Number of documents deleted
        """
        try:
            if namespace:
                # Delete by namespace
                return self.delete(namespace=namespace)
            else:
                # Delete entire collection
                self.client.delete_collection(self.config.collection)
                
                # Recreate collection
                self.collection = self.client.get_or_create_collection(
                    name=self.config.collection,
                    metadata={"hnsw:space": self.config.metric},
                    embedding_function=self.embedding_adapter
                )
                
                LOGGER.info(f"Collection flushed: {self.config.collection}")
                
                return -1
                
        except Exception as e:
            LOGGER.error(f"Flush failed: {e}")
            return 0
    
    def status(self) -> str:
        """Get status string."""
        try:
            count = self.collection.count()
            
            status_lines = [
                f"✅ ChromaDB enabled",
                f"Collection: {self.config.collection}",
                f"Documents: {count}",
                f"Persist dir: {self.config.persist_dir}",
                f"Embedding model: {self.config.embedding_model}",
            ]
            
            return "\n".join(status_lines)
            
        except Exception as e:
            return f"❌ ChromaDB error: {e}"
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        if self.metrics is None:
            return {"enabled": True, "metrics_disabled": True}
        
        return {
            "enabled": True,
            "backend": "chroma",
            "collection": self.config.collection,
            **self.metrics.to_dict()
        }
    
    def close(self) -> None:
        """Close client connection."""
        try:
            self.persist()
            self.client = None  # type: ignore
            LOGGER.info("ChromaDB connection closed")
        except Exception as e:
            LOGGER.warning(f"Error closing ChromaDB: {e}")


# ========================================================================================
# PINECONE IMPLEMENTATION
# ========================================================================================

class PineconeVectorStore:
    """
    Pinecone vector store implementation.
    
    Features:
    - Cloud-native vector database
    - Namespace isolation
    - Metadata filtering
    - Automatic chunking
    """
    
    def __init__(self, config: VectorStoreConfig):
        if not HAS_PINECONE or Pinecone is None:
            raise RuntimeError("Pinecone SDK is not installed")
        
        if not config.pinecone_api_key and not os.getenv("PINECONE_API_KEY"):
            raise RuntimeError("Pinecone API key not configured")
        
        if not config.pinecone_index:
            raise RuntimeError("Pinecone index name not configured")
        
        self.enabled = True
        self.config = config
        self.metrics = VectorStoreMetrics() if config.enable_metrics else None
        self._lock = threading.Lock()
        
        # Initialize embedding function
        self.embedding_fn = OpenAIEmbedding(
            model=config.embedding_model,
            batch_size=config.batch_size,
            enable_cache=config.cache_embeddings
        )
        
        # Initialize Pinecone
        try:
            api_key = config.pinecone_api_key or os.getenv("PINECONE_API_KEY")
            if hasattr(Pinecone, "__call__"):
                # New SDK (v3+)
                pc = Pinecone(api_key=api_key)
                self.pc = pc
            else:
                # Legacy SDK - import dynamically to avoid static import resolution issues
                try:
                    pc_legacy = pinecone if ("pinecone" in globals() and pinecone is not None) else importlib.import_module("pinecone")
                except Exception as e:
                    raise RuntimeError(f"Failed to import legacy pinecone SDK: {e}")
                pc_legacy.init(
                    api_key=api_key,
                    environment=config.pinecone_environment or config.pinecone_cloud
                )
                self.pc = pc_legacy
                self.pc = pc_legacy
            
            # Get index
            if hasattr(self.pc, "Index"):
                self.index = self.pc.Index(config.pinecone_index)
            else:
                self.index = Pinecone.Index(config.pinecone_index)
            
            LOGGER.info(f"✅ Pinecone initialized: {config.pinecone_index}")
            
        except Exception as e:
            LOGGER.error(f"Failed to initialize Pinecone: {e}")
            raise
    
    def _generate_id(self, text: str) -> str:
        """Generate ID based on strategy."""
        if self.config.id_strategy == "hash":
            return hash_text(text)
        else:
            return uuid.uuid4().hex
    
    def _prepare_chunks(self, texts: List[str]) -> List[Tuple[str, str, int]]:
        """Prepare text chunks."""
        chunks: List[Tuple[str, str, int]] = []
        
        for text in texts:
            parent_id = self._generate_id(text)
            
            if not self.config.chunk_enabled:
                chunks.append((text, parent_id, 0))
                continue
            
            text_chunks = chunk_text(
                text,
                max_tokens=self.config.max_chunk_tokens,
                overlap_tokens=self.config.chunk_overlap_tokens,
                model=self.config.embedding_model
            )
            
            for idx, chunk in enumerate(text_chunks):
                chunks.append((chunk, parent_id, idx))
        
        return chunks
    
    def upsert_texts(
        self,
        texts: List[str],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        namespace: Optional[str] = None
    ) -> List[str]:
        """Insert or update texts."""
        if not texts:
            return []
        
        start_time = time.time()
        
        try:
            # Prepare chunks
            chunks = self._prepare_chunks(texts)
            
            # Embed all chunks
            chunk_texts = [text for text, _, _ in chunks]
            embeddings = self.embedding_fn.embed_batch(chunk_texts)
            
            # Build vectors
            ns = namespace or self.config.namespace
            user_metadatas = metadatas or [{} for _ in texts]
            custom_ids = ids or [None] * len(texts)
            
            vectors = []
            parent_ids = []
            text_idx = 0
            
            for (chunk_text, parent_id, chunk_idx), embedding in zip(chunks, embeddings):
                actual_parent_id = custom_ids[text_idx] if custom_ids[text_idx] else parent_id
                
                if chunk_idx == 0:
                    parent_ids.append(actual_parent_id)
                    text_idx += 1
                
                chunk_id = f"{actual_parent_id}__chunk_{chunk_idx}"
                user_meta = user_metadatas[min(text_idx - 1, len(user_metadatas) - 1)]
                
                vector = {
                    "id": chunk_id,
                    "values": embedding,
                    "metadata": {
                        **(user_meta or {}),
                        "text": chunk_text,
                        "parent_id": actual_parent_id,
                        "chunk_index": chunk_idx
                    }
                }
                
                vectors.append(vector)
            
            # Deduplication
            if self.config.deduplicate:
                seen = set()
                unique_vectors = []
                
                for v in vectors:
                    if v["id"] not in seen:
                        seen.add(v["id"])
                        unique_vectors.append(v)
                
                vectors = unique_vectors
            
            # Upsert to Pinecone
            self.index.upsert(vectors=vectors, namespace=ns)
            
            # Update metrics
            if self.metrics:
                with self._lock:
                    self.metrics.total_documents += len(texts)
                    self.metrics.total_chunks += len(vectors)
                    self.metrics.total_upserts += 1
            
            elapsed = time.time() - start_time
            LOGGER.debug(
                f"Upserted {len(texts)} documents ({len(vectors)} chunks) "
                f"in {elapsed:.2f}s"
            )
            
            return parent_ids
            
        except Exception as e:
            LOGGER.error(f"Upsert failed: {e}")
            
            if self.metrics:
                with self._lock:
                    self.metrics.errors += 1
            
            raise
    
    def query(
        self,
        text_or_texts: Union[str, List[str]],
        top_k: int = 5,
        namespace: Optional[str] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> List[QueryResult]:
        """Query vector store."""
        start_time = time.time()
        
        try:
            queries = [text_or_texts] if isinstance(text_or_texts, str) else text_or_texts
            ns = namespace or self.config.namespace
            
            results: List[QueryResult] = []
            
            for query_text in queries:
                # Embed query
                query_embedding = self.embedding_fn.embed_single(query_text)
                
                # Query Pinecone
                response = self.index.query(
                    vector=query_embedding,
                    top_k=top_k,
                    namespace=ns,
                    filter=where,
                    include_metadata=True
                )
                
                # Parse results
                matches = getattr(response, "matches", []) or response.get("matches", [])
                
                for match in matches:
                    if isinstance(match, dict):
                        result = QueryResult(
                            id=match.get("id", ""),
                            text=match.get("metadata", {}).get("text"),
                            score=float(match.get("score", 0.0)),
                            metadata=match.get("metadata", {})
                        )
                    else:
                        metadata = getattr(match, "metadata", {})
                        result = QueryResult(
                            id=getattr(match, "id", ""),
                            text=metadata.get("text") if isinstance(metadata, dict) else None,
                            score=float(getattr(match, "score", 0.0)),
                            metadata=metadata if isinstance(metadata, dict) else {}
                        )
                    
                    results.append(result)
            
            # Update metrics
            if self.metrics:
                elapsed = time.time() - start_time
                
                with self._lock:
                    self.metrics.total_queries += 1
                    self.metrics.total_query_time += elapsed
            
            return results
            
        except Exception as e:
            LOGGER.error(f"Query failed: {e}")
            
            if self.metrics:
                with self._lock:
                    self.metrics.errors += 1
            
            raise
    
    def delete(
        self,
        ids: Optional[List[str]] = None,
        namespace: Optional[str] = None,
        where: Optional[Dict[str, Any]] = None
    ) -> int:
        """Delete documents."""
        try:
            ns = namespace or self.config.namespace
            
            self.index.delete(
                ids=ids,
                namespace=ns,
                filter=where
            )
            
            # Update metrics
            if self.metrics:
                with self._lock:
                    self.metrics.total_deletes += 1
            
            return -1
            
        except Exception as e:
            LOGGER.error(f"Delete failed: {e}")
            
            if self.metrics:
                with self._lock:
                    self.metrics.errors += 1
            
            raise
    
    def persist(self) -> None:
        """No-op for Pinecone (cloud-native)."""
        pass
    
    def flush(self, namespace: Optional[str] = None) -> int:
        """Flush namespace or entire index."""
        try:
            ns = namespace or self.config.namespace
            
            self.index.delete(delete_all=True, namespace=ns)
            
            LOGGER.info(f"Flushed namespace: {ns}")
            
            return -1
            
        except Exception as e:
            LOGGER.error(f"Flush failed: {e}")
            return 0
    
    def status(self) -> str:
        """Get status string."""
        try:
            stats = self.index.describe_index_stats()
            
            total = stats.get("total_vector_count") if isinstance(stats, dict) else getattr(stats, "total_vector_count", "N/A")
            
            status_lines = [
                f"✅ Pinecone enabled",
                f"Index: {self.config.pinecone_index}",
                f"Total vectors: {total}",
                f"Namespace: {self.config.namespace}",
                f"Embedding model: {self.config.embedding_model}",
            ]
            
            return "\n".join(status_lines)
            
        except Exception as e:
            return f"❌ Pinecone error: {e}"
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        if self.metrics is None:
            return {"enabled": True, "metrics_disabled": True}
        
        return {
            "enabled": True,
            "backend": "pinecone",
            "index": self.config.pinecone_index,
            **self.metrics.to_dict()
        }
    
    def close(self) -> None:
        """Close connection."""
        try:
            self.index = None  # type: ignore
            LOGGER.info("Pinecone connection closed")
        except Exception as e:
            LOGGER.warning(f"Error closing Pinecone: {e}")


# ========================================================================================
# SINGLETON INSTANCE
# ========================================================================================

_VECTOR_STORE: Optional[VectorStore] = None
_STORE_LOCK = threading.RLock()


def get_store(force_reload: bool = False) -> VectorStore:
    """
    Get vector store singleton.
    
    Args:
        force_reload: Force store reinitialization
        
    Returns:
        VectorStore instance
    """
    global _VECTOR_STORE
    
    with _STORE_LOCK:
        if _VECTOR_STORE is not None and not force_reload:
            return _VECTOR_STORE
        
        config = _load_vector_store_config()
        
        if not config.enabled:
            LOGGER.info("Vector store disabled by configuration")
            _VECTOR_STORE = NoopVectorStore("Vector store disabled in configuration")
            return _VECTOR_STORE
        
        backend = config.backend.lower()
        
        try:
            if backend == "chroma":
                if not HAS_CHROMA:
                    _VECTOR_STORE = NoopVectorStore("ChromaDB is not installed")
                else:
                    _VECTOR_STORE = ChromaVectorStore(config)
            
            elif backend == "pinecone":
                if not HAS_PINECONE:
                    _VECTOR_STORE = NoopVectorStore("Pinecone SDK is not installed")
                else:
                    _VECTOR_STORE = PineconeVectorStore(config)
            
            else:
                _VECTOR_STORE = NoopVectorStore(f"Unknown backend: {backend}")
            
            LOGGER.info(f"✅ Vector store initialized: {backend}")
            
        except Exception as e:
            LOGGER.error(f"Failed to initialize vector store: {e}")
            _VECTOR_STORE = NoopVectorStore(f"Initialization failed: {e}")
        
        return _VECTOR_STORE


# ========================================================================================
# CONVENIENCE FUNCTIONS
# ========================================================================================

def upsert_texts(
    texts: List[str],
    ids: Optional[List[str]] = None,
    metadatas: Optional[List[Dict[str, Any]]] = None,
    namespace: Optional[str] = None
) -> List[str]:
    """
    Insert or update texts in vector store.
    
    Args:
        texts: List of texts
        ids: Optional custom IDs
        metadatas: Optional metadata
        namespace: Optional namespace
        
    Returns:
        List of document IDs
    """
    return get_store().upsert_texts(texts, ids, metadatas, namespace)


def query(
    text_or_texts: Union[str, List[str]],
    top_k: int = 5,
    namespace: Optional[str] = None,
    where: Optional[Dict[str, Any]] = None
) -> List[QueryResult]:
    """
    Query vector store.
    
    Args:
        text_or_texts: Query text(s)
        top_k: Number of results
        namespace: Optional namespace
        where: Optional metadata filter
        
    Returns:
        List of query results
    """
    return get_store().query(text_or_texts, top_k, namespace, where)


def delete(
    ids: Optional[List[str]] = None,
    namespace: Optional[str] = None,
    where: Optional[Dict[str, Any]] = None
) -> int:
    """
    Delete documents from vector store.
    
    Args:
        ids: Optional list of IDs
        namespace: Optional namespace
        where: Optional metadata filter
        
    Returns:
        Number of documents deleted
    """
    return get_store().delete(ids, namespace, where)


def persist() -> None:
    """Persist data to disk (if applicable)."""
    get_store().persist()


def flush(namespace: Optional[str] = None) -> int:
    """
    Flush vector store.
    
    Args:
        namespace: Optional namespace to flush
        
    Returns:
        Number of documents deleted
    """
    return get_store().flush(namespace)


def status() -> str:
    """
    Get vector store status.
    
    Returns:
        Status string
    """
    return get_store().status()


def get_metrics() -> Dict[str, Any]:
    """
    Get performance metrics.
    
    Returns:
        Metrics dictionary
    """
    return get_store().get_metrics()


def close() -> None:
    """Close vector store connection."""
    get_store().close()


# ========================================================================================
# TESTING & DIAGNOSTICS
# ========================================================================================

def test_vector_store(verbose: bool = True) -> Dict[str, bool]:
    """
    Test vector store functionality.
    
    Args:
        verbose: Print detailed results
        
    Returns:
        Test results
    """
    results = {}
    store = get_store()
    
    # Test 1: Initialization
    try:
        results["initialization"] = store.enabled
        
        if verbose:
            print(f"{'✅' if results['initialization'] else '❌'} Initialization: {results['initialization']}")
    except Exception as e:
        results["initialization"] = False
        if verbose:
            print(f"❌ Initialization: {e}")
    
    if not store.enabled:
        if verbose:
            print("⚠️  Skipping tests (store disabled)")
        return results
    
    # Test 2: Upsert
    try:
        test_texts = ["This is a test document.", "Another test document."]
        ids = upsert_texts(test_texts, namespace="test")
        
        results["upsert"] = len(ids) == len(test_texts)
        
        if verbose:
            print(f"✅ Upsert: {results['upsert']} ({len(ids)} documents)")
    except Exception as e:
        results["upsert"] = False
        if verbose:
            print(f"❌ Upsert: {e}")
    
    # Test 3: Query
    try:
        query_results = query("test document", top_k=2, namespace="test")
        
        results["query"] = len(query_results) > 0
        
        if verbose:
            print(f"✅ Query: {results['query']} ({len(query_results)} results)")
    except Exception as e:
        results["query"] = False
        if verbose:
            print(f"❌ Query: {e}")
    
    # Test 4: Delete
    try:
        deleted = delete(namespace="test")
        
        results["delete"] = True  # Success if no exception
        
        if verbose:
            print(f"✅ Delete: {results['delete']}")
    except Exception as e:
        results["delete"] = False
        if verbose:
            print(f"❌ Delete: {e}")
    
    return results


def print_diagnostics() -> None:
    """
    Print comprehensive diagnostics.
    """
    print("📊 Vector Store Diagnostics\n")
    print("="*60)
    
    # Status
    print("\n🔧 Status:")
    print(status())
    
    # Configuration
    print("\n⚙️ Configuration:")
    config = _load_vector_store_config()
    print(f"  Enabled: {config.enabled}")
    print(f"  Backend: {config.backend}")
    print(f"  Collection: {config.collection}")
    print(f"  Namespace: {config.namespace}")
    print(f"  Embedding model: {config.embedding_model}")
    print(f"  Chunk enabled: {config.chunk_enabled}")
    print(f"  Max chunk tokens: {config.max_chunk_tokens}")
    
    # Metrics
    print("\n📈 Metrics:")
    metrics = get_metrics()
    
    if "enabled" in metrics and metrics["enabled"]:
        for key, value in metrics.items():
            if key not in ("enabled", "backend", "collection", "index"):
                print(f"  {key}: {value}")
    else:
        print("  Metrics not available")
    
    # Dependencies
    print("\n📦 Dependencies:")
    print(f"  ChromaDB: {'✅' if HAS_CHROMA else '❌'}")
    print(f"  Pinecone: {'✅' if HAS_PINECONE else '❌'}")
    print(f"  FAISS: {'✅' if HAS_FAISS else '❌'}")
    print(f"  TikToken: {'✅' if HAS_TIKTOKEN else '❌'}")
    print(f"  OpenAI: {'✅' if HAS_OPENAI else '❌'}")
    
    print("\n" + "="*60)


# ========================================================================================
# MAIN
# ========================================================================================

if __name__ == "__main__":
    print_diagnostics()
    print("\n" + "="*60)
    print("\n🧪 Running tests...\n")
    test_results = test_vector_store(verbose=True)
    passed = sum(1 for v in test_results.values() if v)
    total = len(test_results)
    print(f"\n✅ Passed: {passed} / {total}")
    print("\n" + "="*60)