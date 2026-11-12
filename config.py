import os
from dataclasses import dataclass

def _load_env_file():
    """Tự động đọc file .env ở thư mục hiện tại hoặc thư mục cha và nạp vào os.environ."""
    search_dirs = [
        ".",
        "..",
        os.path.dirname(__file__),
        os.path.dirname(os.path.abspath(__file__)),
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ]
    for d in search_dirs:
        env_path = os.path.join(d, ".env")
        if os.path.isfile(env_path):
            try:
                with open(env_path, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            k, v = line.split("=", 1)
                            k, v = k.strip(), v.strip().strip("'\"")
                            if k and k not in os.environ:
                                os.environ[k] = v
                break
            except Exception:
                pass

_load_env_file()

@dataclass
class SimpleRAGConfig:
    """
    Lớp cấu hình trung tâm cho hệ thống Simple RAG và thực nghiệm đối chiếu ASTUTE RAG.
    Chứa toàn bộ siêu tham số (hyperparameters) cho các tầng: Chunking, Indexing, Retrieval, Generation.
    """
    # 1. Cấu hình Ingestion & Chunking
    chunk_size: int = int(os.getenv("CHUNK_SIZE", "500"))             # Số ký tự tối đa cho mỗi chunk
    chunk_overlap: int = int(os.getenv("CHUNK_OVERLAP", "50"))          # Độ gối đầu giữa 2 chunk liên tiếp

    # 2. Cấu hình Embedder & Vector Store (Hỗ trợ Local Transformers, Ollama và Gemini API)
    embedding_provider: str = os.getenv("EMBEDDING_PROVIDER", "gemini")  # "gemini", "local" (huggingface), "ollama"
    embedding_model_name: str = os.getenv("EMBEDDING_MODEL", "")         # Tên mô hình (vd: BAAI/bge-small-en-v1.5 hoặc models/gemini-embedding-2)
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "0"))            # Số chiều vector (384, 768, 1024, 0 = tự động nhận diện)
    embedding_device: str = os.getenv("EMBEDDING_DEVICE", "cpu")         # Thiết bị chạy local: "cpu" hoặc "cuda"
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434") # Base URL nếu dùng Ollama
    similarity_metric: str = os.getenv("SIMILARITY_METRIC", "cosine")   # Độ đo khoảng cách ("cosine" hoặc "dot_product")
    storage_path: str = os.getenv("STORAGE_PATH", "storage/vector_store.db") # Đường dẫn SQLite lưu trữ chunk ngoài đĩa

    def __post_init__(self):
        self.embedding_provider = (self.embedding_provider or "gemini").lower().strip()
        if not self.embedding_model_name:
            if self.embedding_provider == "gemini":
                self.embedding_model_name = "models/gemini-embedding-2"
            elif self.embedding_provider == "ollama":
                self.embedding_model_name = "nomic-embed-text"
            else:
                self.embedding_model_name = "BAAI/bge-small-en-v1.5"

        if self.embedding_dim <= 0:
            if self.embedding_provider == "gemini":
                self.embedding_dim = 768
            elif self.embedding_provider == "ollama":
                self.embedding_dim = 768
            else:
                m_lower = self.embedding_model_name.lower()
                if any(k in m_lower for k in ["bge-m3", "large"]):
                    self.embedding_dim = 1024
                elif any(k in m_lower for k in ["base", "vietnamese", "768"]):
                    self.embedding_dim = 768
                else:
                    self.embedding_dim = 384

    # 3. Cấu hình Retrieval
    top_k: int = int(os.getenv("TOP_K", "3"))                    # Số lượng chunks truy xuất đưa vào prompt

    # 4. Cấu hình LLM Generation (Hỗ trợ Groq, OpenAI, DeepSeek, Ollama, Gemini Vertex AI...)
    llm_api_key: str = os.getenv("LLM_API_KEY", os.getenv("APIKEY", os.getenv("GEMINI_API_KEY", "")))
    llm_base_url: str = os.getenv("LLM_BASE_URL", os.getenv("BASE_URL", "https://aiplatform.googleapis.com/v1/projects/my-first-rag-514304/locations/global/publishers/google/models"))
    llm_model: str = os.getenv("LLM_MODEL", os.getenv("MODEL", "gemini-3.8-flash"))
    fallback_llm_model: str = os.getenv("FALLBACK_LLM_MODEL", "gemini-3.8-flash")
    max_rpm: int = int(os.getenv("MAX_RPM", "25"))
    temperature: float = float(os.getenv("TEMPERATURE", "0.0"))          # Nhiệt độ sinh từ (0.0: xác định)
    max_tokens: int = int(os.getenv("MAX_TOKENS", "512"))             # Độ dài phản hồi tối đa

    # 5. Cấu hình ASTUTE RAG (Dành cho việc so sánh và khắc phục hạn chế)
    max_internal_passages: int = int(os.getenv("MAX_INTERNAL_PASSAGES", "2"))    # Số đoạn tri thức LLM tự sinh
    num_consolidation_iterations: int = int(os.getenv("NUM_CONSOLIDATION_ITERATIONS", "1")) # Số vòng lặp hợp nhất

    # 6. Cấu hình LLM Judge (Đánh giá RAG Triad - Dùng chung endpoint Generation)
    gemini_api_key: str = os.getenv("GEMINI_API_KEY", os.getenv("APIKEY", ""))
    judge_model: str = os.getenv("JUDGE_MODEL", "gemini-3.8-flash")
    judge_max_rpm: int = int(os.getenv("JUDGE_MAX_RPM", "15"))
    judge_temperature: float = float(os.getenv("JUDGE_TEMPERATURE", "0.0"))
