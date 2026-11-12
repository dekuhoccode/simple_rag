import os
import json
import time
import sqlite3
from typing import List, Optional, Dict, Any
import numpy as np
from schema import Chunk, SearchResult, SourceType

class SimpleVectorStore:
    """
    Giai đoạn 3B: Indexing - Vector Database có khả năng lưu trữ bền vững (Persistent Storage).
    
    Thay vì chỉ lưu trên RAM tạm thời (dễ mất toàn bộ dữ liệu khi bị crash), lớp này:
    - Lưu trữ trực tiếp danh sách Chunk và Embedding Vector vào một file SQLite ngoài đĩa.
    - Chế độ WAL (Write-Ahead Logging) và Transaction giúp ghi nhận tức thì sau mỗi lô chunk,
      đảm bảo không bị corrupt hay mất mát dữ liệu nếu tiến trình dừng đột ngột.
    - Tự động nạp lại (auto-reload) toàn bộ dữ liệu đã lưu từ file khi khởi động lại.
    - Giữ bản cache ma trận vector đã chuẩn hóa L2 trên RAM để truy vấn Cosine Similarity cực nhanh.
    """

    def __init__(self, storage_path: Optional[str] = "storage/vector_store.db"):
        """
        Khởi tạo kho chứa vector với đường dẫn file lưu trữ ngoài đĩa.
        
        Args:
            storage_path: Đường dẫn file SQLite (ví dụ: 'storage/vector_store.db').
                          Nếu là None, sẽ chạy ở chế độ thuần RAM (in-memory).
        """
        self.storage_path = storage_path
        self.chunks: List[Chunk] = []
        self.vectors: Optional[np.ndarray] = None  # Ma trận 2D shape (N, D) trên RAM
        self._conn: Optional[sqlite3.Connection] = None

        if self.storage_path:
            self._init_storage()
            self._load_from_storage()

    def _init_storage(self) -> None:
        """Khởi tạo cấu trúc bảng SQLite và bật chế độ WAL chống crash."""
        dir_name = os.path.dirname(self.storage_path)
        if dir_name:
            os.makedirs(dir_name, exist_ok=True)

        self._conn = sqlite3.connect(self.storage_path, check_same_thread=False)
        with self._conn:
            # WAL mode và synchronous=NORMAL giúp ghi cực nhanh và an toàn trước crash
            self._conn.execute("PRAGMA journal_mode=WAL;")
            self._conn.execute("PRAGMA synchronous=NORMAL;")
            self._conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    doc_id TEXT NOT NULL,
                    content TEXT NOT NULL,
                    source_type TEXT NOT NULL,
                    metadata_json TEXT,
                    embedding BLOB,
                    created_at REAL
                );
            """)

    def _load_from_storage(self) -> None:
        """Đọc toàn bộ chunks và vector đã lưu từ file vào bộ nhớ RAM khi khởi tạo."""
        if not self._conn:
            return

        cursor = self._conn.cursor()
        cursor.execute("SELECT chunk_id, doc_id, content, source_type, metadata_json, embedding FROM chunks ORDER BY created_at ASC")
        rows = cursor.fetchall()

        loaded_chunks: List[Chunk] = []
        loaded_vectors: List[np.ndarray] = []

        for row in rows:
            chunk_id, doc_id, content, source_type_str, meta_json, emb_blob = row
            try:
                source_type = SourceType(source_type_str)
            except Exception:
                source_type = SourceType.EXTERNAL

            meta = json.loads(meta_json) if meta_json else {}
            vec = np.frombuffer(emb_blob, dtype=np.float32) if emb_blob else None

            chunk = Chunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                content=content,
                source_type=source_type,
                metadata=meta,
                embedding=vec
            )
            loaded_chunks.append(chunk)
            if vec is not None:
                loaded_vectors.append(vec)

        self.chunks = loaded_chunks
        if loaded_vectors:
            raw_matrix = np.vstack(loaded_vectors)
            # Chuẩn hóa L2 để sẵn sàng tính Cosine
            norms = np.linalg.norm(raw_matrix, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            self.vectors = (raw_matrix / norms).astype(np.float32)

    @property
    def size(self) -> int:
        """Trả về số lượng chunks hiện có trong kho."""
        return len(self.chunks)

    def add_chunks(self, chunks: List[Chunk], embeddings: Optional[np.ndarray] = None) -> None:
        """
        Nạp danh sách chunk và vector tương ứng vào kho:
        1. Ghi và commit ngay lập tức vào file SQLite ngoài đĩa (tránh mất mát khi crash).
        2. Cập nhật vào danh sách RAM và ma trận vector để phục vụ tìm kiếm nhanh.
        
        Args:
            chunks: Danh sách Chunk cần lưu.
            embeddings: Ma trận 2D chứa vector của các chunk (nếu None sẽ lấy từ c.embedding).
        """
        if not chunks:
            return

        if embeddings is None:
            raw_vecs = [c.embedding for c in chunks if c.embedding is not None]
            if len(raw_vecs) != len(chunks):
                raise ValueError("Tất cả chunk đều phải có vector embedding trước khi nạp vào Vector Store.")
            embeddings = np.vstack(raw_vecs)

        embeddings = embeddings.astype(np.float32)
        now = time.time()

        # 1. Ghi bền vững vào file SQLite (ngay lập tức)
        if self._conn:
            with self._conn:
                for chunk, vec in zip(chunks, embeddings):
                    chunk.embedding = vec
                    meta_json = json.dumps(chunk.metadata, ensure_ascii=False)
                    emb_blob = vec.tobytes()
                    self._conn.execute("""
                        INSERT OR REPLACE INTO chunks 
                        (chunk_id, doc_id, content, source_type, metadata_json, embedding, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        chunk.chunk_id,
                        chunk.doc_id,
                        chunk.content,
                        chunk.source_type.value,
                        meta_json,
                        emb_blob,
                        now
                    ))

        # 2. Cập nhật bộ nhớ RAM (chuẩn hóa L2 để tìm kiếm Cosine tối ưu)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        normalized_new_vectors = (embeddings / norms).astype(np.float32)

        # Xử lý cập nhật đồng bộ trên RAM theo chunk_id (tránh nhân đôi bản ghi khi nạp lại)
        existing_indices = {c.chunk_id: idx for idx, c in enumerate(self.chunks)}
        new_chunks = []
        new_vecs = []

        for c, v in zip(chunks, normalized_new_vectors):
            if c.chunk_id in existing_indices:
                idx = existing_indices[c.chunk_id]
                self.chunks[idx] = c
                if self.vectors is not None:
                    self.vectors[idx] = v
            else:
                new_chunks.append(c)
                new_vecs.append(v)

        if new_chunks:
            self.chunks.extend(new_chunks)
            new_matrix = np.vstack(new_vecs)
            if self.vectors is None:
                self.vectors = new_matrix
            else:
                self.vectors = np.vstack([self.vectors, new_matrix])

    def similarity_search(self, query_vector: np.ndarray, top_k: int = 3) -> List[SearchResult]:
        """
        Tìm kiếm Top-K chunk có độ tương đồng Cosine cao nhất với vector câu hỏi.
        
        Args:
            query_vector: Vector 1 chiều của câu truy vấn (shape: [D,]).
            top_k: Số lượng kết quả gần nhất cần lấy.
            
        Returns:
            List[SearchResult]: Danh sách Top-K kết quả kèm điểm tương đồng và thứ hạng rank (1, 2, 3...).
        """
        if self.vectors is None or len(self.chunks) == 0:
            return []

        # Chuẩn hóa L2 cho query vector
        q = query_vector.astype(np.float32)
        q_norm_val = np.linalg.norm(q)
        if q_norm_val == 0 or np.isnan(q_norm_val):
            return []
        q_norm = q / q_norm_val

        # Cosine similarity = dot product (vì self.vectors đã được chuẩn hóa L2)
        scores = np.dot(self.vectors, q_norm)

        # Lấy top_k chỉ số có điểm cao nhất
        actual_k = min(top_k, len(self.chunks))
        top_indices = np.argsort(scores)[::-1][:actual_k]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            results.append(SearchResult(
                chunk=self.chunks[idx],
                score=float(scores[idx]),
                rank=rank
            ))
        return results

    def clear(self) -> None:
        """Xóa toàn bộ dữ liệu cả trên RAM và trong file SQLite ngoài đĩa."""
        self.chunks.clear()
        self.vectors = None
        if self._conn:
            with self._conn:
                self._conn.execute("DELETE FROM chunks;")

    def close(self) -> None:
        """Đóng kết nối cơ sở dữ liệu nếu có."""
        if self._conn:
            self._conn.close()
            self._conn = None
