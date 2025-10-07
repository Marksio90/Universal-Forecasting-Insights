# src/database/vector_store.py — TURBO PRO (back-compat)
from __future__ import annotations
import os
import time
import json
import math
import hashlib
import pathlib
import uuid
import logging
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

import yaml

# Optional: Streamlit secrets
try:
    import streamlit as st  # type: ignore
except Exception:
    st = None  # pragma: no cover

# --- OpenAI embeddings via shared integrator ---
try:
    from src.ai_engine.openai_integrator import get_client
except Exception:  # fallback if path differs
    try:
        from ai_engine.openai_integrator import get_client  # type: ignore
    except Exception:
        get_client = None  # type: ignore

# --- tiktoken (opcjonalnie, do lepszego chunkowania) ---
try:
    import tiktoken  # type: ignore
    _TIKTOKEN = True
except Exception:
    _TIKTOKEN = False

# --- Chroma (required by requirements.txt) ---
try:
    import chromadb
    from chromadb.api.types import Documents, Embeddings
    from chromadb.utils.embedding_functions import EmbeddingFunction as ChromaEmbeddingFunction
    CHROMA_AVAILABLE = True
except Exception:
    CHROMA_AVAILABLE = False

# --- Pinecone (optional) ---
_PINECONE_AVAILABLE = False
try:
    import pinecone  # type: ignore  # v3+
    _PINECONE_AVAILABLE = True
except Exception:
    try:
        from pinecone import Pinecone  # type: ignore  # legacy shim name
        _PINECONE_AVAILABLE = True
    except Exception:
        _PINECONE_AVAILABLE = False


# =========================================
# Logger
# =========================================
def _get_logger(name: str = "vector_store", level: int = logging.INFO) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        log.setLevel(level)
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                                         datefmt="%Y-%m-%d %H:%M:%S"))
        log.addHandler(h)
        log.propagate = False
    return log

LOGGER = _get_logger()


# =========================================
# Konfiguracja
# =========================================

@dataclass(frozen=True)
class VSOptions:
    enabled: bool = False
    backend: str = "chroma"                 # "chroma" | "pinecone"
    collection: str = "default"
    persist_dir: str = "data/vector_store"
    namespace: str = "intelligent-predictor"
    metric: str = "cosine"                  # "cosine" | "dotproduct" | "euclidean"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536               # 1536(small) / 3072(large)
    batch_size: int = 128

    # PRO: dodatkowe ficzery (opcjonalne)
    # id & dedupe
    id_strategy: str = "uuid"               # "uuid" | "hash"
    deduplicate: bool = False               # jeżeli True i id_strategy="hash" — pomija duplikaty
    # chunking
    chunk_long: bool = True
    max_chunk_tokens: int = 7500            # ~dla text-embedding-3-*
    chunk_overlap_tokens: int = 200
    # pinecone
    pinecone_index: Optional[str] = None
    pinecone_api_key: Optional[str] = None
    pinecone_environment: Optional[str] = None  # legacy
    pinecone_cloud: Optional[str] = None        # nowsze (np. "us-east-1-aws")


def _load_vs_config() -> Dict[str, Any]:
    """
    Ładuje konfigurację wektorowego store z:
      1) config.yaml -> section `vector_store`
      2) st.secrets["vector_store"]
      3) zmienne środowiskowe
    """
    cfg: Dict[str, Any] = {
        "enabled": False,
        "backend": "chroma",
        "collection": "default",
        "persist_dir": "data/vector_store",
        "namespace": "intelligent-predictor",
        "metric": "cosine",
        "embedding_model": "text-embedding-3-small",
        "embedding_dim": 1536,
        "batch_size": 128,
        # PRO:
        "id_strategy": "uuid",
        "deduplicate": False,
        "chunk_long": True,
        "max_chunk_tokens": 7500,
        "chunk_overlap_tokens": 200,
        # Pinecone:
        "pinecone_index": None,
        "pinecone_api_key": None,
        "pinecone_environment": None,
        "pinecone_cloud": None,
    }

    # config.yaml
    try:
        cfg_path = pathlib.Path("config.yaml")
        if cfg_path.exists():
            raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            vs = raw.get("vector_store") or {}
            if isinstance(vs, dict):
                for k in cfg.keys():
                    if k in vs:
                        cfg[k] = vs[k]
    except Exception:
        pass

    # st.secrets
    if st is not None:
        try:
            s = st.secrets.get("vector_store", {})  # type: ignore[attr-defined]
            if s and isinstance(s, dict):
                for k in cfg.keys():
                    if k in s:
                        cfg[k] = s[k]
        except Exception:
            pass

    # env overrides
    env_map = {
        "enabled": ("VECTOR_ENABLED", lambda v: v.lower() in ("1", "true", "yes")),
        "backend": ("VECTOR_BACKEND", str),
        "collection": ("VECTOR_COLLECTION", str),
        "persist_dir": ("VECTOR_PERSIST_DIR", str),
        "namespace": ("VECTOR_NAMESPACE", str),
        "metric": ("VECTOR_METRIC", str),
        "embedding_model": ("OPENAI_EMBED_MODEL", str),
        "embedding_dim": ("OPENAI_EMBED_DIM", int),
        "batch_size": ("VECTOR_BATCH_SIZE", int),
        # PRO:
        "id_strategy": ("VECTOR_ID_STRATEGY", str),
        "deduplicate": ("VECTOR_DEDUPLICATE", lambda v: v.lower() in ("1", "true", "yes")),
        "chunk_long": ("VECTOR_CHUNK_LONG", lambda v: v.lower() in ("1", "true", "yes")),
        "max_chunk_tokens": ("VECTOR_MAX_CHUNK_TOKENS", int),
        "chunk_overlap_tokens": ("VECTOR_CHUNK_OVERLAP", int),
        # Pinecone:
        "pinecone_index": ("PINECONE_INDEX", str),
        "pinecone_api_key": ("PINECONE_API_KEY", str),
        "pinecone_environment": ("PINECONE_ENVIRONMENT", str),
        "pinecone_cloud": ("PINECONE_CLOUD", str),
    }
    for key, (env_key, caster) in env_map.items():
        val = os.getenv(env_key)
        if val is not None:
            try:
                cfg[key] = caster(val)  # type: ignore
            except Exception:
                cfg[key] = val

    return cfg


# =========================================
# Embeddings
# =========================================

def _hash_text(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()

def _token_len(s: str, model_hint: str) -> int:
    if not _TIKTOKEN:
        # przybliżenie: 1 token ~ 4 znaki
        return max(1, math.ceil(len(s) / 4))
    try:
        enc = tiktoken.encoding_for_model(model_hint)  # type: ignore
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")  # type: ignore
    return len(enc.encode(s))  # type: ignore

def _chunk_text(s: str, max_tokens: int, overlap: int, model_hint: str) -> List[str]:
    if not s:
        return []
    if not _TIKTOKEN:
        # fallback po znakach (mniej precyzyjnie)
        approx_char = max_tokens * 4
        ov_char = max(0, overlap * 4)
        if len(s) <= approx_char:
            return [s]
        chunks = []
        start = 0
        while start < len(s):
            end = min(len(s), start + approx_char)
            chunks.append(s[start:end])
            if end == len(s):
                break
            start = max(0, end - ov_char)
        return chunks

    # Dokładniejsze chunkowanie tokenowe
    try:
        enc = tiktoken.encoding_for_model(model_hint)  # type: ignore
    except Exception:
        enc = tiktoken.get_encoding("cl100k_base")  # type: ignore
    toks = enc.encode(s)  # type: ignore
    if len(toks) <= max_tokens:
        return [s]
    chunks = []
    start = 0
    while start < len(toks):
        end = min(len(toks), start + max_tokens)
        chunk = enc.decode(toks[start:end])  # type: ignore
        chunks.append(chunk)
        if end == len(toks):
            break
        start = max(0, end - overlap)
    return chunks


class _OpenAIEmbeddingFn:
    """Batchowy wrapper na OpenAI embeddings API z retry/backoff."""

    def __init__(self, model: str, batch_size: int = 128):
        self.model = model
        self.batch_size = max(1, int(batch_size))

    def embed(self, texts: List[str]) -> List[List[float]]:
        if get_client is None:
            raise RuntimeError("Brak integratora OpenAI (get_client).")
        client = get_client()
        if client is None:
            raise RuntimeError("Brak klucza OpenAI dla embeddings.")

        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            # retry z backoffem
            delay = 0.75
            last_err = None
            for attempt in range(4):
                try:
                    resp = client.embeddings.create(model=self.model, input=chunk)
                    out.extend([d.embedding for d in resp.data])
                    last_err = None
                    break
                except Exception as e:
                    last_err = e
                    time.sleep(delay)
                    delay *= 2.0
            if last_err:
                raise RuntimeError(f"OpenAI embeddings failed after retries: {last_err}")
        return out


class _ChromaEmbeddingAdapter(ChromaEmbeddingFunction):  # type: ignore
    """Adapter EmbeddingFunction dla Chroma."""
    def __init__(self, model: str, batch_size: int = 128):
        self._emb = _OpenAIEmbeddingFn(model=model, batch_size=batch_size)
    def __call__(self, docs: Documents) -> Embeddings:  # type: ignore[override]
        if not docs:
            return []
        texts = list(docs)
        return self._emb.embed(texts)


# =========================================
# Interfejs i Noop
# =========================================

class VectorStoreProtocol:
    enabled: bool
    def upsert_texts(self, texts: List[str], ids: Optional[List[str]] = None,
                     metadatas: Optional[List[Dict[str, Any]]] = None,
                     namespace: Optional[str] = None) -> List[str]: ...
    def query(self, text_or_texts: Union[str, List[str]], top_k: int = 5,
              namespace: Optional[str] = None, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]: ...
    def delete(self, ids: Optional[List[str]] = None, namespace: Optional[str] = None,
               where: Optional[Dict[str, Any]] = None) -> int: ...
    def persist(self) -> None: ...
    def flush(self) -> int: ...
    def status(self) -> str: ...
    def close(self) -> None: ...


@dataclass
class NoopVectorStore(VectorStoreProtocol):
    enabled: bool = False
    reason: str = "Vector store disabled (set vector_store.enabled=true in config.yaml)."

    def upsert_texts(self, texts: List[str], ids=None, metadatas=None, namespace=None) -> List[str]:
        return [ids[i] if ids and i < len(ids) else f"noop-{i}" for i in range(len(texts))]
    def query(self, text_or_texts, top_k=5, namespace=None, where=None) -> List[Dict[str, Any]]:
        return []
    def delete(self, ids=None, namespace=None, where=None) -> int:
        return 0
    def status(self) -> str:
        return self.reason
    def persist(self) -> None:  # pragma: no cover
        pass
    def flush(self) -> int:  # pragma: no cover
        return 0
    def close(self) -> None:  # pragma: no cover
        pass


# =========================================
# Chroma backend
# =========================================

class ChromaVectorStore(VectorStoreProtocol):
    def __init__(self, opts: VSOptions):
        if not CHROMA_AVAILABLE:
            raise RuntimeError("chromadb nie jest zainstalowane.")
        self.enabled = True
        self.opts = opts
        pathlib.Path(opts.persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=opts.persist_dir)
        self.embedder = _ChromaEmbeddingAdapter(model=opts.embedding_model, batch_size=opts.batch_size)
        self.collection = self.client.get_or_create_collection(
            name=opts.collection,
            metadata={"hnsw:space": opts.metric},
            embedding_function=self.embedder,
        )

    def _ns_meta(self, namespace: Optional[str]) -> Dict[str, Any]:
        return {"ns": (namespace or self.opts.namespace)}

    def _gen_id(self, txt: str) -> str:
        if self.opts.id_strategy == "hash":
            return _hash_text(txt)
        return uuid.uuid4().hex

    def _prep_texts(self, texts: List[str]) -> List[Tuple[str, str, int, int]]:
        """
        Zwraca listę (text, parent_id, chunk_id, order).
        Gdy chunkowanie wyłączone — jeden chunk per text (order=0).
        """
        items: List[Tuple[str, str, int, int]] = []
        for t in texts:
            parent_id = self._gen_id(t)
            if not self.opts.chunk_long:
                items.append((t, parent_id, 0, 0))
                continue
            chunks = _chunk_text(
                t,
                max_tokens=max(200, self.opts.max_chunk_tokens),
                overlap=max(0, self.opts.chunk_overlap_tokens),
                model_hint=self.opts.embedding_model,
            )
            for idx, ch in enumerate(chunks):
                items.append((ch, parent_id, idx, idx))
        return items

    def upsert_texts(self, texts: List[str], ids: Optional[List[str]] = None,
                     metadatas: Optional[List[Dict[str, Any]]] = None,
                     namespace: Optional[str] = None) -> List[str]:
        if not texts:
            return []
        # Przygotuj chunki + metadane
        prepared = self._prep_texts(texts)
        base_meta = self._ns_meta(namespace)
        metas_in = metadatas or [{} for _ in texts]
        out_ids: List[str] = []

        # Jeżeli caller podał ręczne `ids` — przypisz je do parentów
        parent_ids: List[str] = ids if ids else [None] * len(texts)  # type: ignore
        parent_idx = 0

        docs: List[str] = []
        ids_final: List[str] = []
        metas_final: List[Dict[str, Any]] = []

        for (chunk_text, parent_hash, chunk_id, order) in prepared:
            # parent_id: preferuj dostarczone ID użytkownika (sekwencyjnie do tekstów)
            par = parent_ids[parent_idx] if parent_idx < len(parent_ids) and parent_ids[parent_idx] else parent_hash
            if order == 0:
                # pierwszy chunk nowego tekstu → przesuwamy indeks rodzica
                parent_idx += 1
                out_ids.append(par)
            # identyfikator chunku (stabilny)
            cid = f"{par}__{chunk_id}"
            meta_user = metas_in[min(parent_idx-1, len(metas_in)-1)] if metas_in else {}
            metas_final.append({**base_meta, **(meta_user or {}), "parent_id": par, "chunk_id": chunk_id, "order": order})
            ids_final.append(cid)
            docs.append(chunk_text)

        # deduplikacja (dla strategii hash): nie wrzucaj chunków o takich samych ID
        if self.opts.deduplicate:
            seen = set()
            uniq_docs, uniq_ids, uniq_meta = [], [], []
            for d, i, m in zip(docs, ids_final, metas_final):
                if i in seen:
                    continue
                seen.add(i)
                uniq_docs.append(d)
                uniq_ids.append(i)
                uniq_meta.append(m)
            docs, ids_final, metas_final = uniq_docs, uniq_ids, uniq_meta

        self.collection.upsert(documents=docs, ids=ids_final, metadatas=metas_final)
        return out_ids

    def query(self, text_or_texts: Union[str, List[str]], top_k: int = 5,
              namespace: Optional[str] = None, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        queries = [text_or_texts] if isinstance(text_or_texts, str) else text_or_texts
        filt = {**self._ns_meta(namespace), **(where or {})}
        res = self.collection.query(query_texts=queries, n_results=top_k, where=filt)
        results: List[Dict[str, Any]] = []
        ids = res.get("ids") or []
        for i in range(len(ids)):
            for j, _id in enumerate(ids[i]):
                results.append(
                    {
                        "id": _id,
                        "text": (res.get("documents", [[]])[i][j] if res.get("documents") else None),
                        "score": (res.get("distances", [[]])[i][j] if res.get("distances") else None),
                        "metadata": (res.get("metadatas", [[]])[i][j] if res.get("metadatas") else None),
                    }
                )
        return results

    def delete(self, ids: Optional[List[str]] = None, namespace: Optional[str] = None,
               where: Optional[Dict[str, Any]] = None) -> int:
        filt = {**self._ns_meta(namespace), **(where or {})}
        self.collection.delete(ids=ids, where=filt if filt else None)
        # Chroma nie zwraca count — sygnalizujemy -1
        return -1

    def persist(self) -> None:
        try:
            self.client.persist()
        except Exception:
            pass

    def flush(self) -> int:
        try:
            self.client.delete_collection(self.opts.collection)
            self.collection = self.client.get_or_create_collection(
                name=self.opts.collection,
                metadata={"hnsw:space": self.opts.metric},
                embedding_function=self.embedder,
            )
            return -1
        except Exception:
            return 0

    def status(self) -> str:
        try:
            n = self.collection.count()
            return (f"ChromaDB enabled • collection={self.opts.collection} • count={n} "
                    f"• persist_dir={self.opts.persist_dir} • model={self.opts.embedding_model}")
        except Exception as e:
            return f"ChromaDB error: {e}"

    def close(self) -> None:
        try:
            self.client = None  # type: ignore
        except Exception:
            pass


# =========================================
# Pinecone backend (opcjonalny)
# =========================================

class PineconeVectorStore(VectorStoreProtocol):
    def __init__(self, opts: VSOptions):
        if not _PINECONE_AVAILABLE:
            raise RuntimeError("Pinecone SDK nie jest zainstalowany.")
        self.enabled = True
        self.opts = opts
        self.embedder = _OpenAIEmbeddingFn(model=opts.embedding_model, batch_size=opts.batch_size)

        # Inicjalizacja klienta
        try:
            if hasattr(pinecone, "Pinecone"):
                pc = pinecone.Pinecone(api_key=(opts.pinecone_api_key or os.getenv("PINECONE_API_KEY")))  # type: ignore[attr-defined]
                self.pc = pc
            else:
                pinecone.init(api_key=(opts.pinecone_api_key or os.getenv("PINECONE_API_KEY")),
                              environment=opts.pinecone_environment or opts.pinecone_cloud)  # type: ignore
                self.pc = pinecone  # type: ignore
        except Exception as e:
            raise RuntimeError(f"Pinecone init error: {e}")

        # Upewnij się, że index istnieje
        try:
            if hasattr(self.pc, "list_indexes"):
                names = [i["name"] for i in self.pc.list_indexes()]  # type: ignore[index]
            else:
                names = [i.name for i in self.pc.list_indexes().indexes]  # type: ignore[attr-defined]
        except Exception:
            names = []
        if not opts.pinecone_index:
            raise RuntimeError("Brak nazwy indeksu Pinecone (pinecone_index).")
        if opts.pinecone_index not in names:
            try:
                if hasattr(self.pc, "create_index"):
                    self.pc.create_index(  # type: ignore[attr-defined]
                        name=opts.pinecone_index,
                        dimension=int(opts.embedding_dim),
                        metric=opts.metric if opts.metric != "cosine" else "cosine",
                    )
                else:
                    self.pc.create_index(name=opts.pinecone_index, dimension=int(opts.embedding_dim), metric=opts.metric)
            except Exception as e:
                LOGGER.warning("Pinecone create_index failed (może istnieć lub brak uprawnień): %s", e)

        # Uchwyt do indeksu
        if hasattr(self.pc, "Index"):
            self.index = self.pc.Index(opts.pinecone_index)  # type: ignore[attr-defined]
        else:
            self.index = pinecone.Index(opts.pinecone_index)  # type: ignore

    def _ns(self, namespace: Optional[str]) -> str:
        return namespace or self.opts.namespace

    def _prep_texts(self, texts: List[str]) -> List[Tuple[str, str, int, int]]:
        items: List[Tuple[str, str, int, int]] = []
        for t in texts:
            parent_id = uuid.uuid4().hex if self.opts.id_strategy != "hash" else _hash_text(t)
            if not self.opts.chunk_long:
                items.append((t, parent_id, 0, 0))
                continue
            chunks = _chunk_text(
                t,
                max_tokens=max(200, self.opts.max_chunk_tokens),
                overlap=max(0, self.opts.chunk_overlap_tokens),
                model_hint=self.opts.embedding_model,
            )
            for idx, ch in enumerate(chunks):
                items.append((ch, parent_id, idx, idx))
        return items

    def upsert_texts(self, texts: List[str], ids: Optional[List[str]] = None,
                     metadatas: Optional[List[Dict[str, Any]]] = None,
                     namespace: Optional[str] = None) -> List[str]:
        if not texts:
            return []

        prepared = self._prep_texts(texts)
        # embed w batchach
        chunk_texts = [t for (t, _, _, _) in prepared]
        embs = self.embedder.embed(chunk_texts)

        metas_in = metadatas or [{} for _ in texts]
        ns = self._ns(namespace)

        vectors = []
        out_parent_ids: List[str] = []
        parent_ids_given = ids if ids else [None] * len(texts)  # type: ignore
        parent_idx = 0

        for (chunk_text, parent_hash, chunk_id, order), vec in zip(prepared, embs):
            par = parent_ids_given[parent_idx] if parent_idx < len(parent_ids_given) and parent_ids_given[parent_idx] else parent_hash
            if order == 0:
                parent_idx += 1
                out_parent_ids.append(par)
            cid = f"{par}__{chunk_id}"
            meta_user = metas_in[min(parent_idx-1, len(metas_in)-1)] if metas_in else {}
            vectors.append({"id": cid, "values": vec,
                            "metadata": {**(meta_user or {}), "text": chunk_text,
                                         "parent_id": par, "chunk_id": chunk_id, "order": order}})

        if self.opts.deduplicate:
            seen = set()
            uniq = []
            for v in vectors:
                if v["id"] in seen:
                    continue
                seen.add(v["id"])
                uniq.append(v)
            vectors = uniq

        # upsert
        # Pinecone v3: upsert(vectors=[...], namespace=...)
        self.index.upsert(vectors=vectors, namespace=ns)
        return out_parent_ids

    def query(self, text_or_texts: Union[str, List[str]], top_k: int = 5,
              namespace: Optional[str] = None, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        queries = [text_or_texts] if isinstance(text_or_texts, str) else text_or_texts
        results: List[Dict[str, Any]] = []
        ns = self._ns(namespace)
        for q in queries:
            q_emb = self.embedder.embed([q])[0]
            resp = self.index.query(
                vector=q_emb, top_k=top_k, include_values=False, include_metadata=True,
                namespace=ns, filter=where,
            )
            matches = getattr(resp, "matches", []) or resp.get("matches", [])  # type: ignore
            for m in matches:
                # dict vs object kompatybilnie
                if isinstance(m, dict):
                    results.append({"id": m.get("id"),
                                    "text": (m.get("metadata") or {}).get("text"),
                                    "score": m.get("score"),
                                    "metadata": (m.get("metadata") or {})})
                else:
                    results.append({"id": getattr(m, "id", None),
                                    "text": getattr(getattr(m, "metadata", {}), "get", lambda k, d=None: None)("text"),
                                    "score": getattr(m, "score", None),
                                    "metadata": getattr(m, "metadata", {})})
        return results

    def delete(self, ids: Optional[List[str]] = None, namespace: Optional[str] = None,
               where: Optional[Dict[str, Any]] = None) -> int:
        try:
            self.index.delete(ids=ids, namespace=self._ns(namespace), filter=where)
            return -1
        except Exception:
            return 0

    def persist(self) -> None:
        pass

    def flush(self) -> int:
        try:
            self.index.delete(delete_all=True, namespace=self._ns(None))
            return -1
        except Exception:
            return 0

    def status(self) -> str:
        try:
            stats = self.index.describe_index_stats()
            total = stats.get("total_vector_count") if isinstance(stats, dict) else getattr(stats, "total_vector_count", "n/a")
            return (f"Pinecone enabled • index={self.opts.pinecone_index} • vectors={total} "
                    f"• ns={self.opts.namespace} • model={self.opts.embedding_model}")
        except Exception as e:
            return f"Pinecone error: {e}"

    def close(self) -> None:
        try:
            self.index = None  # type: ignore
        except Exception:
            pass


# =========================================
# Fabryka / singleton
# =========================================

_STORE_SINGLETON: Optional[VectorStoreProtocol] = None

def _opts_from_cfg(cfg: Dict[str, Any]) -> VSOptions:
    return VSOptions(
        enabled=bool(cfg.get("enabled", False)),
        backend=str(cfg.get("backend", "chroma")),
        collection=str(cfg.get("collection", "default")),
        persist_dir=str(cfg.get("persist_dir", "data/vector_store")),
        namespace=str(cfg.get("namespace", "intelligent-predictor")),
        metric=str(cfg.get("metric", "cosine")),
        embedding_model=str(cfg.get("embedding_model", "text-embedding-3-small")),
        embedding_dim=int(cfg.get("embedding_dim", 1536)),
        batch_size=int(cfg.get("batch_size", 128)),
        id_strategy=str(cfg.get("id_strategy", "uuid")),
        deduplicate=bool(cfg.get("deduplicate", False)),
        chunk_long=bool(cfg.get("chunk_long", True)),
        max_chunk_tokens=int(cfg.get("max_chunk_tokens", 7500)),
        chunk_overlap_tokens=int(cfg.get("chunk_overlap_tokens", 200)),
        pinecone_index=cfg.get("pinecone_index"),
        pinecone_api_key=cfg.get("pinecone_api_key"),
        pinecone_environment=cfg.get("pinecone_environment"),
        pinecone_cloud=cfg.get("pinecone_cloud"),
    )

def get_store() -> VectorStoreProtocol:
    global _STORE_SINGLETON
    if _STORE_SINGLETON is not None:
        return _STORE_SINGLETON

    cfg = _load_vs_config()
    opts = _opts_from_cfg(cfg)
    if not opts.enabled:
        _STORE_SINGLETON = NoopVectorStore()
        return _STORE_SINGLETON

    backend = opts.backend.lower().strip()
    try:
        if backend == "chroma":
            if not CHROMA_AVAILABLE:
                _STORE_SINGLETON = NoopVectorStore(reason="ChromaDB nie jest zainstalowane.")
            else:
                _STORE_SINGLETON = ChromaVectorStore(opts)
        elif backend == "pinecone":
            if not _PINECONE_AVAILABLE:
                _STORE_SINGLETON = NoopVectorStore(reason="Pinecone SDK nie jest zainstalowany.")
            else:
                if not (opts.pinecone_api_key or os.getenv("PINECONE_API_KEY")) or not opts.pinecone_index:
                    _STORE_SINGLETON = NoopVectorStore(reason="Brak PINECONE_API_KEY lub pinecone_index w konfiguracji.")
                else:
                    _STORE_SINGLETON = PineconeVectorStore(opts)
        else:
            _STORE_SINGLETON = NoopVectorStore(reason=f"Nieznany backend vector_store: {backend}")
    except Exception as e:
        LOGGER.exception("Vector store init error")
        _STORE_SINGLETON = NoopVectorStore(reason=f"Vector store init error: {e}")

    return _STORE_SINGLETON


# =========================================
# Wygodne funkcje modułowe (back-compat)
# =========================================

def upsert_texts(texts: List[str], ids: Optional[List[str]] = None,
                 metadatas: Optional[List[Dict[str, Any]]] = None,
                 namespace: Optional[str] = None) -> List[str]:
    store = get_store()
    return store.upsert_texts(texts, ids=ids, metadatas=metadatas, namespace=namespace)

def query(text_or_texts: Union[str, List[str]], top_k: int = 5,
          namespace: Optional[str] = None, where: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    store = get_store()
    return store.query(text_or_texts, top_k=top_k, namespace=namespace, where=where)

def delete(ids: Optional[List[str]] = None, namespace: Optional[str] = None,
           where: Optional[Dict[str, Any]] = None) -> int:
    store = get_store()
    return store.delete(ids=ids, namespace=namespace, where=where)

def persist() -> None:
    store = get_store()
    store.persist()

def flush() -> int:
    store = get_store()
    return store.flush()

def status() -> str:
    store = get_store()
    return store.status()

def close() -> None:
    store = get_store()
    store.close()
