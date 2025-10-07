# src/database/vector_store.py
from __future__ import annotations
import os
import time
import json
import pathlib
import uuid
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
    from ai_engine.openai_integrator import get_client  # type: ignore

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
    # new SDK
    import pinecone  # type: ignore
    _PINECONE_AVAILABLE = True
except Exception:
    try:
        from pinecone import Pinecone  # type: ignore
        _PINECONE_AVAILABLE = True
    except Exception:
        _PINECONE_AVAILABLE = False


# =========================================
# Konfiguracja
# =========================================

def _load_vs_config() -> Dict[str, Any]:
    """
    Ładuje konfigurację wektorowego store z:
      1) config.yaml -> section `vector_store`
      2) st.secrets["vector_store"]
      3) zmienne środowiskowe
    """
    cfg: Dict[str, Any] = {
        "enabled": False,
        "backend": "chroma",             # "chroma" | "pinecone"
        "collection": "default",
        "persist_dir": "data/vector_store",
        "namespace": "intelligent-predictor",
        "metric": "cosine",              # "cosine" | "dotproduct" | "euclidean"
        "embedding_model": "text-embedding-3-small",
        "embedding_dim": 1536,           # OPENAI dims: 1536(small) / 3072(large)
        "batch_size": 128,
        # Pinecone specyficzne:
        "pinecone_index": None,
        "pinecone_api_key": None,
        "pinecone_environment": None,    # dla starszych instalacji
        "pinecone_cloud": None,          # dla nowszych (us-east-1-aws itp.)
    }

    # config.yaml
    try:
        cfg_path = pathlib.Path("config.yaml")
        if cfg_path.exists():
            raw = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
            vs = raw.get("vector_store") or {}
            if isinstance(vs, dict):
                cfg.update({k: vs.get(k, cfg[k]) for k in cfg.keys() if k in vs})
    except Exception:
        pass

    # st.secrets
    if st is not None:
        try:
            s = st.secrets.get("vector_store", {})  # type: ignore[attr-defined]
            if s:
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

class _OpenAIEmbeddingFn:
    """Prosty batchowy wrapper na OpenAI embeddings API."""

    def __init__(self, model: str, batch_size: int = 128):
        self.model = model
        self.batch_size = batch_size

    def embed(self, texts: List[str]) -> List[List[float]]:
        client = get_client()
        if client is None:
            raise RuntimeError("Brak klucza OpenAI dla embeddings.")
        out: List[List[float]] = []
        for i in range(0, len(texts), self.batch_size):
            chunk = texts[i : i + self.batch_size]
            resp = client.embeddings.create(model=self.model, input=chunk)
            out.extend([d.embedding for d in resp.data])
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

    def upsert_texts(
        self,
        texts: List[str],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        namespace: Optional[str] = None,
    ) -> List[str]:
        raise NotImplementedError

    def query(
        self,
        text_or_texts: Union[str, List[str]],
        top_k: int = 5,
        namespace: Optional[str] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        raise NotImplementedError

    def delete(
        self,
        ids: Optional[List[str]] = None,
        namespace: Optional[str] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> int:
        raise NotImplementedError

    def persist(self) -> None:
        pass

    def flush(self) -> int:
        return 0

    def status(self) -> str:
        return "Vector store disabled."

    def close(self) -> None:
        pass


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


# =========================================
# Chroma backend
# =========================================

class ChromaVectorStore(VectorStoreProtocol):
    def __init__(
        self,
        collection: str,
        persist_dir: str,
        metric: str,
        embed_model: str,
        batch_size: int,
        namespace: str,
    ):
        if not CHROMA_AVAILABLE:
            raise RuntimeError("chromadb nie jest zainstalowane.")
        self.enabled = True
        self.collection_name = collection
        self.persist_dir = persist_dir
        self.metric = metric
        self.namespace = namespace
        pathlib.Path(persist_dir).mkdir(parents=True, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.embedder = _ChromaEmbeddingAdapter(model=embed_model, batch_size=batch_size)
        self.collection = self.client.get_or_create_collection(
            name=self.collection_name,
            metadata={"hnsw:space": metric},
            embedding_function=self.embedder,
        )

    def _ns(self, namespace: Optional[str]) -> Dict[str, Any]:
        ns = namespace or self.namespace
        # Chroma nie ma natywnego namespace; dodamy do metadanych filtr "ns"
        return {"ns": ns}

    def upsert_texts(
        self,
        texts: List[str],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        namespace: Optional[str] = None,
    ) -> List[str]:
        if not texts:
            return []
        ids = ids or [uuid.uuid4().hex for _ in texts]
        base_meta = self._ns(namespace)
        metas = metadatas or [{} for _ in texts]
        metas = [{**base_meta, **(m or {})} for m in metas]
        self.collection.upsert(documents=texts, ids=ids, metadatas=metas)
        return ids

    def query(
        self,
        text_or_texts: Union[str, List[str]],
        top_k: int = 5,
        namespace: Optional[str] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        queries = [text_or_texts] if isinstance(text_or_texts, str) else text_or_texts
        filt = {**self._ns(namespace), **(where or {})}
        res = self.collection.query(query_texts=queries, n_results=top_k, where=filt)
        # Chroma zwraca tablice wyników per zapytanie
        results: List[Dict[str, Any]] = []
        for i in range(len(res.get("ids", []))):
            for j, _id in enumerate(res["ids"][i]):
                results.append(
                    {
                        "id": _id,
                        "text": (res["documents"][i][j] if res.get("documents") else None),
                        "score": res["distances"][i][j] if res.get("distances") else None,
                        "metadata": res["metadatas"][i][j] if res.get("metadatas") else None,
                    }
                )
        return results

    def delete(
        self,
        ids: Optional[List[str]] = None,
        namespace: Optional[str] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> int:
        filt = {**self._ns(namespace), **(where or {})}
        # Chroma: delete returns None; nie ma liczby skasowanych -> zwrócimy -1 gdy unknown
        self.collection.delete(ids=ids, where=filt if filt else None)
        return -1

    def persist(self) -> None:
        # PersistentClient zapisuje automatycznie; zostawiamy hook
        try:
            self.client.persist()
        except Exception:
            pass

    def flush(self) -> int:
        # Brak natywnego flush kolekcji; można skasować i odtworzyć
        try:
            count = -1
            self.client.delete_collection(self.collection_name)
            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": self.metric},
                embedding_function=self.embedder,
            )
            return count
        except Exception:
            return 0

    def status(self) -> str:
        try:
            n = self.collection.count()
            return f"ChromaDB enabled • collection={self.collection_name} • count={n} • persist_dir={self.persist_dir}"
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
    def __init__(
        self,
        index_name: str,
        api_key: str,
        metric: str,
        embed_model: str,
        batch_size: int,
        namespace: str,
        environment: Optional[str] = None,
        cloud: Optional[str] = None,
        embedding_dim: Optional[int] = 1536,
    ):
        if not _PINECONE_AVAILABLE:
            raise RuntimeError("Pinecone SDK nie jest zainstalowany.")
        self.enabled = True
        self.index_name = index_name
        self.namespace = namespace
        self.metric = metric
        self.batch_size = batch_size
        self.embedder = _OpenAIEmbeddingFn(model=embed_model, batch_size=batch_size)
        self.embedding_dim = int(embedding_dim or 1536)

        # Inicjalizacja
        try:
            # nowe SDK
            if hasattr(pinecone, "Pinecone"):
                pc = pinecone.Pinecone(api_key=api_key)  # type: ignore[attr-defined]
                self.pc = pc
            else:
                pinecone.init(api_key=api_key, environment=environment or cloud)  # type: ignore
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
        if self.index_name not in names:
            # spróbuj utworzyć (wymaga uprawnień)
            try:
                if hasattr(self.pc, "create_index"):
                    self.pc.create_index(  # type: ignore[attr-defined]
                        name=self.index_name,
                        dimension=self.embedding_dim,
                        metric=self.metric if self.metric != "cosine" else "cosine",
                    )
                else:
                    self.pc.create_index(name=self.index_name, dimension=self.embedding_dim, metric=self.metric)
            except Exception:
                # pomiń – może istnieje już na innym projekcie
                pass

        # Pobierz uchwyt do indeksu
        if hasattr(self.pc, "Index"):
            self.index = self.pc.Index(self.index_name)  # type: ignore[attr-defined]
        else:
            self.index = pinecone.Index(self.index_name)  # type: ignore

    def _ns(self, namespace: Optional[str]) -> str:
        return namespace or self.namespace

    def upsert_texts(
        self,
        texts: List[str],
        ids: Optional[List[str]] = None,
        metadatas: Optional[List[Dict[str, Any]]] = None,
        namespace: Optional[str] = None,
    ) -> List[str]:
        if not texts:
            return []
        ids = ids or [uuid.uuid4().hex for _ in texts]
        embs = self.embedder.embed(texts)
        vectors = []
        metas = metadatas or [{} for _ in texts]
        for _id, vec, m, txt in zip(ids, embs, metas, texts):
            item = {"id": _id, "values": vec, "metadata": {**(m or {}), "text": txt}}
            vectors.append(item)

        self.index.upsert(vectors=vectors, namespace=self._ns(namespace))
        return ids

    def query(
        self,
        text_or_texts: Union[str, List[str]],
        top_k: int = 5,
        namespace: Optional[str] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        queries = [text_or_texts] if isinstance(text_or_texts, str) else text_or_texts
        results: List[Dict[str, Any]] = []
        for q in queries:
            q_emb = self.embedder.embed([q])[0]
            resp = self.index.query(
                vector=q_emb,
                top_k=top_k,
                include_values=False,
                include_metadata=True,
                namespace=self._ns(namespace),
                filter=where,
            )
            # unify format
            matches = getattr(resp, "matches", []) or resp.get("matches", [])  # type: ignore
            for m in matches:
                results.append(
                    {
                        "id": m["id"] if isinstance(m, dict) else m.id,
                        "text": (m["metadata"].get("text") if isinstance(m, dict) else (m.metadata or {}).get("text")),
                        "score": m["score"] if isinstance(m, dict) else m.score,
                        "metadata": m["metadata"] if isinstance(m, dict) else (m.metadata or {}),
                    }
                )
        return results

    def delete(
        self,
        ids: Optional[List[str]] = None,
        namespace: Optional[str] = None,
        where: Optional[Dict[str, Any]] = None,
    ) -> int:
        try:
            self.index.delete(ids=ids, namespace=self._ns(namespace), filter=where)
            return -1
        except Exception:
            return 0

    def persist(self) -> None:
        # zarządzane przez Pinecone – brak akcji
        pass

    def flush(self) -> int:
        # usuń cały namespace
        try:
            self.index.delete(delete_all=True, namespace=self._ns(None))
            return -1
        except Exception:
            return 0

    def status(self) -> str:
        try:
            stats = self.index.describe_index_stats()
            total = stats.get("total_vector_count") if isinstance(stats, dict) else getattr(stats, "total_vector_count", "n/a")
            return f"Pinecone enabled • index={self.index_name} • vectors={total} • ns={self.namespace}"
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

def get_store() -> VectorStoreProtocol:
    global _STORE_SINGLETON
    if _STORE_SINGLETON is not None:
        return _STORE_SINGLETON

    cfg = _load_vs_config()
    if not cfg.get("enabled", False):
        _STORE_SINGLETON = NoopVectorStore()
        return _STORE_SINGLETON

    backend = str(cfg.get("backend", "chroma")).lower()
    try:
        if backend == "chroma":
            if not CHROMA_AVAILABLE:
                _STORE_SINGLETON = NoopVectorStore(reason="ChromaDB nie jest zainstalowane.")
            else:
                _STORE_SINGLETON = ChromaVectorStore(
                    collection=str(cfg.get("collection", "default")),
                    persist_dir=str(cfg.get("persist_dir", "data/vector_store")),
                    metric=str(cfg.get("metric", "cosine")),
                    embed_model=str(cfg.get("embedding_model", "text-embedding-3-small")),
                    batch_size=int(cfg.get("batch_size", 128)),
                    namespace=str(cfg.get("namespace", "intelligent-predictor")),
                )
        elif backend == "pinecone":
            if not _PINECONE_AVAILABLE:
                _STORE_SINGLETON = NoopVectorStore(reason="Pinecone SDK nie jest zainstalowany.")
            else:
                api_key = cfg.get("pinecone_api_key") or os.getenv("PINECONE_API_KEY")
                index_name = cfg.get("pinecone_index")
                if not api_key or not index_name:
                    _STORE_SINGLETON = NoopVectorStore(reason="Brak PINECONE_API_KEY lub pinecone_index w konfiguracji.")
                else:
                    _STORE_SINGLETON = PineconeVectorStore(
                        index_name=str(index_name),
                        api_key=str(api_key),
                        metric=str(cfg.get("metric", "cosine")),
                        embed_model=str(cfg.get("embedding_model", "text-embedding-3-small")),
                        batch_size=int(cfg.get("batch_size", 128)),
                        namespace=str(cfg.get("namespace", "intelligent-predictor")),
                        environment=cfg.get("pinecone_environment"),
                        cloud=cfg.get("pinecone_cloud"),
                        embedding_dim=int(cfg.get("embedding_dim", 1536)),
                    )
        else:
            _STORE_SINGLETON = NoopVectorStore(reason=f"Nieznany backend vector_store: {backend}")
    except Exception as e:
        _STORE_SINGLETON = NoopVectorStore(reason=f"Vector store init error: {e}")

    return _STORE_SINGLETON


# =========================================
# Wygodne funkcje modułowe
# =========================================

def upsert_texts(
    texts: List[str],
    ids: Optional[List[str]] = None,
    metadatas: Optional[List[Dict[str, Any]]] = None,
    namespace: Optional[str] = None,
) -> List[str]:
    store = get_store()
    return store.upsert_texts(texts, ids=ids, metadatas=metadatas, namespace=namespace)

def query(
    text_or_texts: Union[str, List[str]],
    top_k: int = 5,
    namespace: Optional[str] = None,
    where: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    store = get_store()
    return store.query(text_or_texts, top_k=top_k, namespace=namespace, where=where)

def delete(
    ids: Optional[List[str]] = None,
    namespace: Optional[str] = None,
    where: Optional[Dict[str, Any]] = None,
) -> int:
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
