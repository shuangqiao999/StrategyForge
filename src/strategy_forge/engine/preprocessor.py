"""Deduction Engine Preprocessor — semantic chunking + LanceDB indexing + hybrid retrieval.

Embedding calls use synchronous HTTP (requests); LLM entity discovery uses async.
"""
from __future__ import annotations

import asyncio
import logging
import re
import threading
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import requests

logger = logging.getLogger(__name__)


@dataclass
class PreprocessResult:
    session_id: str
    chunks: list[Any]
    high_freq_entities: dict[str, set[str]]
    low_freq_entities: dict[str, set[str]]
    entity_aliases: dict[str, set[str]] = field(default_factory=dict)
    total_chunks: int = 0
    total_entities: int = 0
    entity_frequencies: dict[str, int] = field(default_factory=dict)
    entity_chunk_coverage: dict[str, int] = field(default_factory=dict)


class DeductionPreprocessor:
    # 嵌入文本前缀长度：适配 2048-token 上下文窗口的嵌入模型（如 text-embedding-embeddinggemma-300m-qat）。
    # 中文最坏情况下 ~2 tokens/char，1000 字 ≈ 2000 tokens，安全余量内。
    _INDEX_PREFIX_LEN = 1000

    def __init__(self, workspace_root: str | Path, session_id: str) -> None:
        ws = Path(workspace_root)
        self.workspace_root = ws
        self.session_id = session_id
        self.table_name = f"deduction_chunks_{session_id}"

        self._db: Any = None
        self._table: Any = None
        self._event_table: Any = None
        self._event_table_name: str = ""
        self._dim: int = 0
        self._result: PreprocessResult | None = None

        # 检索加速缓存：chunks 表 preprocess 后不可变 + agent 名/查询高度重复，
        # 故缓存"查询文本→向量"与"实体召回结果"，在优化器 M×N 并发共享同一
        # preprocessor 时把重复嵌入/检索降到近零，避免压垮本地嵌入服务。
        self._cache_lock = threading.Lock()
        self._embed_cache: dict[str, list[float]] = {}
        self._recall_cache: dict[tuple, list[str]] = {}
        self._dynamic_cache: dict[tuple, list[str]] = {}
        self._fts_ready: bool = False
        self._event_fts_ready: bool = False
        self._event_fts_dirty: bool = False
        self._progress_cb: Any = None

        self.embed_cache_hits: int = 0
        self.recall_cache_hits: int = 0

        self._embed_config = self._resolve_embed_config()
        self._embed_url = self._resolve_embed_url()
        self._embed_model = self._resolve_embed_model()
        self._http = requests.Session()
        self._http.headers["Content-Type"] = "application/json"
        api_key = self._embed_config.get("api_key", "") or ""
        if api_key:
            self._http.headers["Authorization"] = f"Bearer {api_key}"

        self._init_lancedb()

    def _resolve_embed_config(self) -> dict:
        """Resolve embedding config via the unified provider registry."""
        try:
            from strategy_forge.core.providers import registry
            return registry.resolve_for_embedding()
        except Exception as e:
            logger.warning("[Preprocessor] Failed to read embedding config: %s", e)
            return {}

    def _resolve_embed_url(self) -> str:
        base = self._embed_config.get("api_base", "") or ""
        if base:
            return base.rstrip("/") + "/embeddings"
        logger.warning("[Preprocessor] No embedding_api_base configured — "
                       "LanceDB indexing will be skipped")
        return ""

    def _resolve_embed_model(self) -> str:
        return self._embed_config.get("model_name", "") or ""

    def _init_lancedb(self) -> None:
        import lancedb
        from strategy_forge.core.config import config
        lance_dir = str(config.deduction_data_dir / "lancedb")
        Path(lance_dir).mkdir(parents=True, exist_ok=True)
        self._db = lancedb.connect(lance_dir)

    def _create_or_open(self, name: str, schema: Any) -> Any:
        """健壮地获取一张 LanceDB 表：存在则打开，不存在则创建。

        防止重跑/上次崩溃残留导致 table_names() 未列出但 create(mode="create")
        又报 'already exists' 的冲突：create 失败时回退 open；open 再失败则覆盖重建。
        """
        try:
            if name in self._db.table_names():
                return self._db.open_table(name)
        except Exception as e:
            logger.debug("[Preprocessor] table_names/open probe failed for %s: %s", name, e)
        try:
            return self._db.create_table(name, schema=schema, mode="create")
        except Exception as e:
            logger.warning("[Preprocessor] create '%s' failed (%s); 尝试打开已存在表", name, e)
            try:
                return self._db.open_table(name)
            except Exception as e2:
                logger.warning("[Preprocessor] open '%s' 也失败 (%s)，覆盖重建", name, e2)
                return self._db.create_table(name, schema=schema, mode="overwrite")

    def _ensure_table(self, dim: int) -> None:
        if self._table is not None:
            return
        import pyarrow as pa
        schema = pa.schema([
            pa.field("id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("content", pa.string()),
            pa.field("session_id", pa.string()),
        ])
        self._table = self._create_or_open(self.table_name, schema)

    def _ensure_event_table(self, dim: int) -> None:
        if self._event_table is not None:
            return
        import pyarrow as pa
        schema = pa.schema([
            pa.field("event_id", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), dim)),
            pa.field("content", pa.string()),
            pa.field("agent_id", pa.string()),
            pa.field("round_number", pa.int32()),
            pa.field("session_id", pa.string()),
            pa.field("priority", pa.float32()),
            pa.field("event_type", pa.string()),
            pa.field("visibility", pa.string()),
            pa.field("participants", pa.string()),
        ])
        self._event_table_name = f"deduction_events_{self.session_id}"
        self._event_table = self._create_or_open(self._event_table_name, schema)

    # ── Sync Embedding (no asyncio) ──

    def _sync_embed_single(self, text: str) -> list[float]:
        key = text[:self._INDEX_PREFIX_LEN]
        with self._cache_lock:
            cached = self._embed_cache.get(key)
            if cached is not None:
                self.embed_cache_hits += 1
                return cached
        r = self._http.post(self._embed_url, json={
            "input": key,
            "model": self._embed_model,
        }, timeout=60)
        r.raise_for_status()
        vec = r.json()["data"][0]["embedding"]
        with self._cache_lock:
            self._embed_cache[key] = vec
        return vec

    def _sync_embed_batch(self, texts: list[str]) -> list[list[float]]:
        r = self._http.post(self._embed_url, json={
            "input": [t[:self._INDEX_PREFIX_LEN] for t in texts],
            "model": self._embed_model,
        }, timeout=120)
        r.raise_for_status()
        return [d["embedding"] for d in r.json()["data"]]

    def _auto_detect_dim(self) -> int:
        try:
            vec = self._sync_embed_single("dimension auto-detect probe")
            if vec and len(vec) > 0:
                logger.info("[Preprocessor] Auto-detected embedding dimension: %d", len(vec))
                return len(vec)
        except Exception as e:
            logger.warning("[Preprocessor] Dimension probe failed: %s", e)
        return 0

    # ── Dynamic event memory ──

    def add_event_memory(self, content: str, agent_id: str,
                         round_number: int, event_type: str = "",
                         priority: float = 0.5,
                         visibility: str = "public",
                         participants: str = "") -> None:
        if self._event_table is None or self._dim <= 0:
            return
        embed_text = f"[R{round_number}] {event_type}: {content}"[:self._INDEX_PREFIX_LEN]
        try:
            vec = self._sync_embed_single(embed_text)
        except Exception as e:
            logger.debug("[Preprocessor] add_event_memory embed failed: %s", e)
            return
        try:
            self._event_table.add([{
                "event_id": str(uuid.uuid4()),
                "vector": vec, "content": content,
                "agent_id": agent_id, "round_number": round_number,
                "session_id": self.session_id,
                "priority": priority, "event_type": event_type,
                "visibility": visibility or "public",
                "participants": participants or "",
            }])
        except Exception as e1:
            logger.warning("[Preprocessor] event add failed (full schema): %s, trying reduced schema", e1)
            try:
                # Fallback for tables without visibility/participants columns
                self._event_table.add([{
                    "event_id": str(uuid.uuid4()),
                    "vector": vec, "content": content,
                    "agent_id": agent_id, "round_number": round_number,
                    "session_id": self.session_id,
                    "priority": priority, "event_type": event_type,
                }])
            except Exception as e2:
                logger.warning("[Preprocessor] event add failed (reduced schema): %s, trying minimal schema", e2)
                try:
                    # Fallback for old tables without priority/event_type columns
                    self._event_table.add([{
                        "event_id": str(uuid.uuid4()),
                        "vector": vec, "content": content,
                        "agent_id": agent_id, "round_number": round_number,
                        "session_id": self.session_id,
                    }])
                except Exception as e3:
                    logger.error("[Preprocessor] event add failed (all schemas): %s — event data lost", e3)
        # 事件表已变更，标记 FTS 索引需在下次检索前重建（每轮至多一次）
        self._event_fts_dirty = True

    def retrieve_latest_intervention(self) -> dict | None:
        """检索最近的用户干预或不可变目标指令。

        优先用 LanceDB 的 .where() 过滤下推（避免整表 to_arrow 扫描）；不支持时回退全表扫描。
        """
        if self._event_table is None:
            return None
        where_clause = ("priority >= 0.9 OR "
                        "event_type IN ('user_intervention', 'immutable_goal')")
        try:
            rows = self._event_table.search().where(where_clause).limit(100).to_list()
            interventions = [{
                "content": r.get("content", ""),
                "round_number": r.get("round_number", 0) or 0,
                "priority": r.get("priority", 0.0) or 0.0,
            } for r in rows]
            if interventions:
                interventions.sort(key=lambda x: (-x["priority"], -x["round_number"]))
                return interventions[0]
            return None
        except Exception:
            return self._intervention_scan()

    def _intervention_scan(self) -> dict | None:
        """回退路径：全表扫描筛选干预/目标（旧实现，兼容不支持 where 的环境）。"""
        try:
            raw = self._event_table.to_arrow().to_pydict()
            has_priority = "priority" in raw
            has_etype = "event_type" in raw
            interventions = []
            for i in range(len(raw["event_id"])):
                p = raw.get("priority", [0])[i] if has_priority else 0
                et = raw.get("event_type", [""])[i] if has_etype else ""
                if p >= 0.9 or et in ("user_intervention", "immutable_goal"):
                    interventions.append({
                        "content": raw["content"][i],
                        "round_number": raw["round_number"][i],
                        "priority": p,
                    })
            if interventions:
                interventions.sort(key=lambda x: (-x["priority"], -x.get("round_number", 0)))
                return interventions[0]
        except Exception:
            pass
        return None

    _EVENT_EXCLUDED_TYPES = frozenset({"immutable_goal", "user_intervention"})

    def _ensure_event_fts(self) -> None:
        """按需为动态事件表建/重建 content 全文索引（混合检索用）。

        事件增量写入后置 _event_fts_dirty；此处每轮至多重建一次，控制开销。
        原生 FTS（lancedb 0.33，无需 tantivy）；失败则保持纯向量检索。
        """
        if self._event_table is None:
            return
        if self._event_fts_ready and not self._event_fts_dirty:
            return
        try:
            self._event_table.create_fts_index("content", replace=True)
            self._event_fts_ready = True
            self._event_fts_dirty = False
            logger.debug("[Preprocessor] Rebuilt event FTS index (hybrid enabled)")
        except Exception as e:
            self._event_fts_ready = False
            logger.debug("[Preprocessor] event FTS skipped (vector-only): %s", e)

    def retrieve_dynamic_events(
        self, query_text: str, top_k: int = 3, min_similarity: float = 0.4,
        observer: str = "",
    ) -> list[str]:
        """检索动态事件记忆。

        observer 非空时启用可见性过滤：visibility=private 的事件仅当
        observer（名称或实体ID）出现在 participants 或为事件行动者时可见。
        observer 为空 = 上帝视角（报告器/裁判使用），可见全部事件。
        """
        if self._event_table is None or self._dim <= 0:
            return []
        from strategy_forge.core.providers import registry as _reg
        use_hybrid = _reg.event_hybrid
        cache_key = (query_text[:80], top_k, min_similarity, use_hybrid, observer)
        cached = self._dynamic_cache.get(cache_key)
        if cached is not None:
            return cached
        try:
            query_vec = self._sync_embed_single(query_text)
        except Exception:
            return []
        if use_hybrid:
            self._ensure_event_fts()
        hybrid = use_hybrid and self._event_fts_ready and bool(query_text)
        fetch_k = top_k * 3
        where_clause = "event_type NOT IN ('immutable_goal', 'user_intervention')"
        try:
            if hybrid:
                q = (self._event_table.search(query_type="hybrid")
                     .vector(query_vec).text(query_text))
            else:
                q = self._event_table.search(query_vec).metric("cosine")
            try:
                q = q.where(where_clause)
            except Exception:
                pass
            raw = q.limit(fetch_k).to_list()
        except Exception:
            # 混合检索若因索引/版本异常失败，兜底回退纯向量
            if hybrid:
                try:
                    raw = (self._event_table.search(query_vec).metric("cosine")
                           .limit(fetch_k).to_list())
                    hybrid = False
                except Exception:
                    return []
            else:
                return []
        if not raw:
            return []
        min_distance = 1.0 - min_similarity
        results: list[str] = []
        for r in raw:
            # Python 侧强制剔除目标/干预事件，保证即使 hybrid 的 where 未生效也不泄漏
            if r.get("event_type", "") in self._EVENT_EXCLUDED_TYPES:
                continue
            # 可见性过滤：私密事件只有参与者可见（信息差是博弈资源）
            if observer and (r.get("visibility", "") or "public") == "private":
                parts = r.get("participants", "") or ""
                if observer not in parts and observer != (r.get("agent_id", "") or ""):
                    continue
            # hybrid: 靠 RRF 融合排序 + top_k 截断，不套用余弦阈值；
            # 纯向量: 沿用 _distance 相似度门槛过滤。
            if not hybrid and r.get("_distance", 10.0) >= min_distance:
                continue
            content = r.get("content", "")
            if content and content not in results:
                results.append(content[:300])
            if len(results) >= top_k:
                break
        self._dynamic_cache[cache_key] = results
        return results

    def clear_round_cache(self) -> None:
        self._dynamic_cache.clear()
    def retrieve_for_entity(
        self, entity_name: str, top_k: int = 5,
        must_contain: set[str] | None = None,
    ) -> list[str]:
        if self._table is None or self._dim <= 0:
            return []
        cache_key = (entity_name, top_k, frozenset(must_contain) if must_contain else None)
        with self._cache_lock:
            cached = self._recall_cache.get(cache_key)
            if cached is not None:
                self.recall_cache_hits += 1
                return list(cached)
        try:
            query_vec = self._sync_embed_single(entity_name)
        except Exception:
            return []
        try:
            raw = self._hybrid_or_vector_search(self._table, query_vec, entity_name, top_k * 3)
        except Exception:
            return []
        results: list[str] = []
        min_dist = getattr(config, "deduction_similarity_threshold", 0.3)
        for r in raw:
            dist = r.get("_distance", 0.0)
            if dist > min_dist:
                continue
            content = r.get("content", "")
            if not content:
                continue
            if must_contain and not any(kw in content for kw in must_contain):
                continue
            if content not in results:
                results.append(content)
            if len(results) >= top_k:
                break
        with self._cache_lock:
            self._recall_cache[cache_key] = list(results)
        return results

    def _hybrid_or_vector_search(self, table, query_vec, query_text, fetch_k):
        if self._fts_ready and query_text:
            q = table.search(query_type="hybrid").vector(query_vec).text(query_text)
        else:
            q = table.search(query_vec).metric("cosine")
        return q.limit(fetch_k).to_list()

    # ── Main preprocessing pipeline ──

    @property
    def result(self) -> PreprocessResult | None:
        return self._result

    async def preprocess(self, source: str) -> PreprocessResult:
        from strategy_forge.core.chunker import TextChunker
        from strategy_forge.core.config import config as _cfg
        from strategy_forge.core.tokenizer import extract_named_entities

        # 1. semantic chunking
        _cs = max(256, _cfg.deduction_chunk_size)
        chunker = TextChunker(strategy="paragraph", max_chunk_size=_cs)
        chunks = chunker.chunk(source, file_type=".txt")
        chunk_texts = [c.content for c in chunks]
        logger.info("[Preprocessor] Chunked into %d semantic chunks", len(chunks))

        # 2. Jieba POS entity extraction
        all_entities = extract_named_entities(source, top_k=1000, min_freq=1)
        high_freq: dict[str, set[str]] = {}
        low_freq: dict[str, set[str]] = {}
        entity_freq: dict[str, int] = {}
        entity_chunk_cov: dict[str, int] = {}
        for std_name, aliases in all_entities.items():
            count = len(re.findall(re.escape(std_name), source))
            for alias in aliases:
                if alias != std_name and alias not in std_name:
                    count += len(re.findall(re.escape(alias), source))
            entity_freq[std_name] = count
            entity_chunk_cov[std_name] = sum(
                1 for ct in chunk_texts
                if std_name in ct or any(a in ct for a in aliases))
            if count >= 2:
                high_freq[std_name] = aliases
            else:
                low_freq[std_name] = aliases
        logger.info("[Preprocessor] Entities (jieba): %d total, %d high-freq, %d low-freq",
                    len(all_entities), len(high_freq), len(low_freq))

        # 3. LanceDB vector indexing
        dim = self._auto_detect_dim()
        self._dim = dim
        if dim <= 0:
            logger.warning("[Preprocessor] Dimension is 0, skipping LanceDB indexing")
            self._result = PreprocessResult(
                session_id=self.session_id, chunks=list(chunks),
                high_freq_entities=high_freq, low_freq_entities=low_freq,
                total_chunks=len(chunks), total_entities=len(all_entities),
                entity_frequencies=entity_freq, entity_chunk_coverage=entity_chunk_cov)
            return self._result

        self._ensure_table(dim)
        self._ensure_event_table(dim)

        chunk_ids = [f"chunk-{uuid.uuid4().hex[:8]}" for _ in chunks]
        chunk_prefixes = [c.content[:self._INDEX_PREFIX_LEN] for c in chunks]
        try:
            vecs = self._sync_embed_batch(chunk_prefixes)
        except Exception as e:
            logger.warning("[Preprocessor] Batch embed failed: %s", e)
            self._result = PreprocessResult(
                session_id=self.session_id, chunks=list(chunks),
                high_freq_entities=high_freq, low_freq_entities=low_freq,
                entity_aliases=all_entities,
                total_chunks=len(chunks), total_entities=len(all_entities),
                entity_frequencies=entity_freq, entity_chunk_coverage=entity_chunk_cov)
            return self._result

        rows = [{"id": chunk_ids[i], "vector": vecs[i],
                 "content": chunks[i].content, "session_id": self.session_id}
                for i in range(len(chunks))]
        self._table.add(rows)
        self._maybe_create_vector_index(self._table, len(rows))
        # 为静态切片建全文索引，启用混合检索(向量+BM25)；events 表增量追加，不建 FTS。
        try:
            self._table.create_fts_index("content", replace=True)
            self._fts_ready = True
            logger.info("[Preprocessor] Created FTS index on chunks.content (hybrid search enabled)")
        except Exception as e:
            logger.debug("[Preprocessor] FTS index skipped (fallback to vector-only): %s", e)
        logger.info("[Preprocessor] LanceDB indexed %d chunks (dim=%d)", len(rows), dim)

        self._result = PreprocessResult(
            session_id=self.session_id, chunks=list(chunks),
            high_freq_entities=high_freq, low_freq_entities=low_freq,
            entity_aliases=all_entities,
            total_chunks=len(chunks), total_entities=len(all_entities),
            entity_frequencies=entity_freq, entity_chunk_coverage=entity_chunk_cov)
        return self._result

    @staticmethod
    def _maybe_create_vector_index(table: Any, n_rows: int) -> None:
        """数据量足够大时为向量列建 IVF 索引以加速检索；
        小数据集 (< 256 行) LanceDB 暴力 KNN 更快且 IVF 训练样本不足，跳过。"""
        if n_rows < 256:
            return
        try:
            table.create_index(metric="cosine", vector_column_name="vector")
            logger.info("[Preprocessor] Created LanceDB vector index (rows=%d)", n_rows)
        except Exception as e:
            logger.debug("[Preprocessor] Vector index skipped: %s", e)

    def set_progress_callback(self, cb: Any) -> None:
        """Set callback for long-running phases. Called as cb(current, total)."""
        self._progress_cb = cb

    def close(self) -> None:
        self._http.close()
        self._table = None
        self._db = None

    def drop_tables(self) -> None:
        """物理删除当前会话的 LanceDB 表，回收磁盘空间。"""
        if self._db is None:
            return
        patterns = (f"deduction_chunks_{self.session_id}", f"deduction_events_{self.session_id}")
        for name in self._db.table_names():
            if name in patterns:
                try:
                    self._db.drop_table(name)
                    logger.info("[Preprocessor] Dropped table: %s", name)
                except Exception as e:
                    logger.warning("[Preprocessor] Failed to drop %s: %s", name, e)
