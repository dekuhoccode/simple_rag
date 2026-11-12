import os
import json
import time
import random
import urllib.request
import urllib.error
from typing import List, Optional, Dict, Any
import numpy as np
from schema import Chunk

class SimpleEmbedder:
    """
    Giai đoạn 3A: Indexing - Vector Embedding đa nguồn (Multi-provider).
    
    Hỗ trợ 3 chế độ linh hoạt:
    1. 'gemini': Sử dụng Google Gemini Embedding REST API (mặc định: models/gemini-embedding-2).
    2. 'local': Chạy hoàn toàn trên máy cục bộ bằng HuggingFace Transformers hoặc SentenceTransformers
       (Ví dụ: 'BAAI/bge-small-en-v1.5', 'sentence-transformers/all-MiniLM-L6-v2', 'BAAI/bge-m3').
       Ưu điểm: Miễn phí, không tốn quota API, chạy nhanh trên CPU hoặc GPU.
    3. 'ollama': Sử dụng máy chủ Ollama chạy local (Ví dụ: 'nomic-embed-text').
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"
    MAX_BATCH_SIZE = 50  # Mặc định cho Gemini

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        output_dimensionality: int = 768,
        task_prefix: str = "search result",
        provider: Optional[str] = None,
        device: str = "cpu",
        ollama_base_url: str = "http://localhost:11434",
        batch_size: Optional[int] = None
    ):
        """
        Khởi tạo Embedder đa phương thức.
        
        Args:
            model_name: Tên mô hình (vd: 'models/gemini-embedding-2' hoặc 'BAAI/bge-small-en-v1.5').
            api_key: Khóa API (cho Gemini nếu dùng provider='gemini').
            output_dimensionality: Số chiều vector (Mặc định 768 hoặc 384 tùy model).
            task_prefix: Tiền tố tác vụ cho Gemini Embedding ('search result' hoặc 'question answering').
            provider: 'gemini', 'local' (huggingface), hoặc 'ollama'. Tự động phát hiện nếu để None.
            device: Thiết bị chạy local: 'cpu' hoặc 'cuda'.
            ollama_base_url: URL máy chủ Ollama nếu dùng provider='ollama'.
            batch_size: Kích thước mẻ xử lý khi embed chunks.
        """
        # 1. Xác định Provider
        if provider:
            self.provider = provider.lower().strip()
        else:
            env_provider = os.getenv("EMBEDDING_PROVIDER", "").lower().strip()
            if env_provider:
                self.provider = env_provider
            elif model_name and (model_name.startswith("models/") or "gemini" in model_name):
                self.provider = "gemini"
            elif model_name and any(k in model_name.lower() for k in ["bge", "minilm", "sentence", "bert", "roberta", "vietnamese"]):
                self.provider = "local"
            else:
                self.provider = "gemini"

        # 2. Xác định Model Name theo Provider
        if not model_name:
            if self.provider == "gemini":
                model_name = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2")
            elif self.provider == "ollama":
                model_name = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")
            else:
                model_name = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

        if self.provider == "gemini":
            self.model_name = model_name if model_name.startswith("models/") else f"models/{model_name}"
            self.max_batch_size = batch_size or 50
        else:
            self.model_name = model_name
            self.max_batch_size = batch_size or 32

        self.api_key = api_key or os.getenv("GEMINI_API_KEY", os.getenv("APIKEY", ""))
        self.output_dimensionality = output_dimensionality
        self.task_prefix = task_prefix
        self.device = device or os.getenv("EMBEDDING_DEVICE", "cpu")
        self.ollama_base_url = ollama_base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        self.is_v2_model = "gemini-embedding-2" in self.model_name

        # Thuộc tính lazy-load cho local model backend
        self._local_initialized = False
        self._local_backend = None
        self._st_model = None
        self._hf_tokenizer = None
        self._hf_model = None
        self._torch = None

    # =========================================================================
    # PHẦN A: BACKEND LOCAL TRANSFORMERS / SENTENCE-TRANSFORMERS
    # =========================================================================
    def _init_local_model(self):
        """Khởi tạo mô hình nhúng cục bộ (chạy trên CPU hoặc GPU)."""
        if self._local_initialized:
            return

        # 1. Thử dùng thư viện sentence-transformers nếu người dùng đã cài
        try:
            from sentence_transformers import SentenceTransformer
            print(f"[*] Đang tải mô hình local '{self.model_name}' bằng SentenceTransformer ({self.device})...")
            self._st_model = SentenceTransformer(self.model_name, device=self.device)
            self._local_backend = "sentence_transformers"
            self._local_initialized = True
            return
        except ImportError:
            pass

        # 2. Sử dụng thư viện transformers & torch có sẵn trong môi trường
        try:
            import torch
            from transformers import AutoTokenizer, AutoModel
            print(f"[*] Đang tải mô hình local '{self.model_name}' bằng HuggingFace Transformers ({self.device})...")
            self._torch = torch
            self._hf_tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self._hf_model = AutoModel.from_pretrained(self.model_name)
            self._hf_model.eval()
            self._hf_model.to(self.device)
            self._local_backend = "transformers"
            self._local_initialized = True
        except Exception as e:
            raise RuntimeError(
                f"[Embedder] Không thể tải mô hình embedding local '{self.model_name}'.\n"
                f"Lỗi: {e}\n"
                f"Gợi ý: Đảm bảo máy có kết nối internet ở lần chạy đầu tiên để tải weights, "
                f"hoặc kiểm tra 'pip install transformers torch'."
            )

    def _mean_pooling(self, model_output, attention_mask):
        """Tính trung bình có trọng số (mean pooling) có áp dụng attention mask."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        sum_embeddings = self._torch.sum(token_embeddings * input_mask_expanded, 1)
        sum_mask = self._torch.clamp(input_mask_expanded.sum(1), min=1e-9)
        return sum_embeddings / sum_mask

    def _embed_local_texts(self, texts: List[str]) -> np.ndarray:
        """Thực hiện nhúng danh sách text thành vector numpy float32 bằng local model."""
        if not texts:
            return np.empty((0, self.output_dimensionality or 384), dtype=np.float32)

        self._init_local_model()

        # Dùng SentenceTransformers nếu có
        if self._local_backend == "sentence_transformers":
            vecs = self._st_model.encode(
                texts,
                batch_size=self.max_batch_size,
                show_progress_bar=False,
                normalize_embeddings=True
            )
            return np.array(vecs, dtype=np.float32)

        # Dùng HuggingFace Transformers chuẩn
        all_vecs = []
        for i in range(0, len(texts), self.max_batch_size):
            batch_texts = texts[i:i + self.max_batch_size]
            encoded = self._hf_tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt"
            ).to(self.device)

            with self._torch.no_grad():
                model_output = self._hf_model(**encoded)
                pooled = self._mean_pooling(model_output, encoded["attention_mask"])
                # Chuẩn hóa L2 để tương thích tốt nhất với độ đo Cosine
                normalized = self._torch.nn.functional.normalize(pooled, p=2, dim=1)
                all_vecs.append(normalized.cpu().numpy().astype(np.float32))

        return np.vstack(all_vecs)

    # =========================================================================
    # PHẦN B: BACKEND OLLAMA LOCAL
    # =========================================================================
    def _embed_ollama_texts(self, texts: List[str]) -> np.ndarray:
        """Gửi request tới API Ollama chạy local."""
        if not texts:
            return np.empty((0, self.output_dimensionality or 768), dtype=np.float32)

        base_url = self.ollama_base_url.rstrip("/")
        # Thử endpoint batch /api/embed (Ollama v0.1.34+)
        embed_url = f"{base_url}/api/embed"
        payload = {"model": self.model_name, "input": texts}
        try:
            req = urllib.request.Request(
                embed_url,
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=120) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                if "embeddings" in data:
                    return np.array(data["embeddings"], dtype=np.float32)
        except Exception:
            pass

        # Fallback về endpoint cũ /api/embeddings (từng câu)
        legacy_url = f"{base_url}/api/embeddings"
        vecs = []
        for t in texts:
            p = {"model": self.model_name, "prompt": t}
            req = urllib.request.Request(
                legacy_url,
                data=json.dumps(p).encode("utf-8"),
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                vecs.append(data.get("embedding", []))
        return np.array(vecs, dtype=np.float32)

    # =========================================================================
    # PHẦN C: BACKEND GOOGLE GEMINI API
    # =========================================================================
    def _format_document_text(self, chunk: Chunk) -> str:
        """Định dạng văn bản chunk theo chuẩn Asymmetric Retrieval của Gemini Embedding."""
        title = (chunk.metadata or {}).get("title", "none")
        return f"title: {title} | text: {chunk.content.strip()}"

    def _format_query_text(self, query: str) -> str:
        """Định dạng câu hỏi truy vấn theo chuẩn Asymmetric Retrieval của Gemini Embedding."""
        return f"task: {self.task_prefix} | query: {query.strip()}"

    def _post_api(self, endpoint_suffix: str, payload: Dict[str, Any], max_retries: int = 5) -> Dict[str, Any]:
        """Gửi request HTTP POST tới Gemini REST API qua urllib kèm retry."""
        url = f"{self.BASE_URL}/{self.model_name}:{endpoint_suffix}?key={self.api_key}"
        data_bytes = json.dumps(payload).encode("utf-8")

        for attempt in range(max_retries):
            req = urllib.request.Request(
                url,
                data=data_bytes,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8", errors="ignore")
                if e.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    wait_time = (5.0 * (2 ** attempt)) + random.uniform(1.0, 3.0)
                    print(f"\n[Embedder HTTP {e.code}] Quá tải rate limit. Đợi {wait_time:.1f}s trước khi thử lại (lần {attempt + 1}/{max_retries})...")
                    time.sleep(wait_time)
                    continue
                print(f"\n[Embedder Lỗi HTTP {e.code}]: {err_body[:300]}")
                raise e
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(3.0 * (attempt + 1))
                    continue
                raise e
        return {}

    # =========================================================================
    # PHẦN D: CÁC PHƯƠNG THỨC PUBLIC CHÍNH (CHUẨN HOÁ CHO TOÀN BỘ RAG PIPELINE)
    # =========================================================================
    def embed_text(self, text: str) -> np.ndarray:
        """Chuyển một chuỗi văn bản đơn lẻ thành vector 1 chiều (1D numpy array float32)."""
        if self.provider == "local":
            vecs = self._embed_local_texts([text])
            return vecs[0]
        elif self.provider == "ollama":
            vecs = self._embed_ollama_texts([text])
            return vecs[0]
        else:
            payload = {
                "model": self.model_name,
                "content": {"parts": [{"text": text}]},
                "outputDimensionality": self.output_dimensionality
            }
            data = self._post_api("embedContent", payload)
            values = data.get("embedding", {}).get("values", [])
            return np.array(values, dtype=np.float32)

    def embed_chunks(self, chunks: List[Chunk]) -> np.ndarray:
        """
        Nhúng danh sách các Chunk văn bản thành ma trận 2D (N, D).
        Gán trực tiếp vector tương ứng vào thuộc tính chunk.embedding của từng chunk.
        """
        if not chunks:
            dim = self.output_dimensionality or 768
            return np.empty((0, dim), dtype=np.float32)

        # 1. Chạy với Local Provider
        if self.provider == "local":
            texts = [c.content for c in chunks]
            vectors = self._embed_local_texts(texts)
            for chunk, vec in zip(chunks, vectors):
                chunk.embedding = vec
            if self.output_dimensionality and vectors.shape[1] != self.output_dimensionality:
                self.output_dimensionality = vectors.shape[1]
            return vectors

        # 2. Chạy với Ollama Provider
        elif self.provider == "ollama":
            texts = [c.content for c in chunks]
            vectors = self._embed_ollama_texts(texts)
            for chunk, vec in zip(chunks, vectors):
                chunk.embedding = vec
            if self.output_dimensionality and vectors.shape[1] != self.output_dimensionality:
                self.output_dimensionality = vectors.shape[1]
            return vectors

        # 3. Chạy với Gemini Provider (theo lô 50)
        all_vectors = []
        for start_idx in range(0, len(chunks), self.max_batch_size):
            batch = chunks[start_idx: start_idx + self.max_batch_size]
            requests_payload = []
            for chunk in batch:
                formatted = self._format_document_text(chunk) if self.is_v2_model else chunk.content
                requests_payload.append({
                    "model": self.model_name,
                    "content": {"parts": [{"text": formatted}]},
                    "outputDimensionality": self.output_dimensionality
                })

            data = self._post_api("batchEmbedContents", {"requests": requests_payload})
            embeddings_list = data.get("embeddings", data.get("embedding", []))

            for chunk, emb_obj in zip(batch, embeddings_list):
                vec = np.array(emb_obj.get("values", []), dtype=np.float32)
                chunk.embedding = vec
                all_vectors.append(vec)

        if not all_vectors:
            return np.empty((0, self.output_dimensionality), dtype=np.float32)
        return np.vstack(all_vectors)

    def embed_query(self, query: str) -> np.ndarray:
        """
        Nhúng câu hỏi (truy vấn) của người dùng thành vector để so khớp ngữ nghĩa.
        Áp dụng định dạng tiền tố tương ứng với từng mô hình.
        """
        cleaned = query.strip()
        if self.provider == "gemini":
            formatted = self._format_query_text(cleaned) if self.is_v2_model else cleaned
            return self.embed_text(formatted)
        elif self.provider == "local":
            # Với một số model như BGE-base/large, query có thể cần query instruction
            if "bge" in self.model_name.lower() and "bge-m3" not in self.model_name.lower() and "small" not in self.model_name.lower():
                cleaned = f"Represent this sentence for searching relevant passages: {cleaned}"
            return self.embed_text(cleaned)
        else:
            return self.embed_text(cleaned)
