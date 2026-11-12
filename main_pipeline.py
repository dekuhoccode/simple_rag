from typing import List, Optional
from config import SimpleRAGConfig
from schema import Document, Chunk, SearchResult, GenerationOutput
from ingestion.document_loader import SimpleDocumentLoader
from chunking.text_chunker import SimpleTextChunker
from indexing.embedder import SimpleEmbedder
from indexing.vector_store import SimpleVectorStore
from retrieval.retriever import SimpleRetriever
from generation.prompt_builder import SimplePromptBuilder
from generation.llm_generator import SimpleLLMGenerator

class SimpleRAGPipeline:
    """
    Luồng thực thi hoàn chỉnh End-to-End của mô hình Standard / Simple RAG:
    1. Ingestion: Nạp tài liệu văn bản.
    2. Chunking: Cắt tài liệu thành các đoạn cố định (fixed-size).
    3. Indexing: Nhúng vector và lưu vào Vector Store in-memory.
    4. Retrieval: Tìm Top-K đoạn tương đồng ngữ nghĩa bằng Cosine.
    5. Generation: Nối phẳng các đoạn vào Prompt và gọi LLM sinh câu trả lời.
    """

    def __init__(self, config: Optional[SimpleRAGConfig] = None):
        """Khởi tạo toàn bộ các module thành phần theo cấu hình."""
        self.config = config or SimpleRAGConfig()

        self.loader = SimpleDocumentLoader()
        self.chunker = SimpleTextChunker(
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )
        self.embedder = SimpleEmbedder(
            provider=getattr(self.config, "embedding_provider", "gemini"),
            model_name=self.config.embedding_model_name,
            api_key=getattr(self.config, "gemini_api_key", None),
            output_dimensionality=getattr(self.config, "embedding_dim", 768),
            device=getattr(self.config, "embedding_device", "cpu"),
            ollama_base_url=getattr(self.config, "ollama_base_url", "http://localhost:11434")
        )
        storage_path = getattr(self.config, "storage_path", "storage/vector_store.db")
        self.vector_store = SimpleVectorStore(storage_path=storage_path)
        self.retriever = SimpleRetriever(
            embedder=self.embedder,
            vector_store=self.vector_store,
            top_k=self.config.top_k
        )
        self.prompt_builder = SimplePromptBuilder()
        self.generator = SimpleLLMGenerator(
            model_name=self.config.llm_model,
            api_key=getattr(self.config, "llm_api_key", None),
            base_url=getattr(self.config, "llm_base_url", None),
            fallback_model_name=getattr(self.config, "fallback_llm_model", None),
            max_rpm=getattr(self.config, "max_rpm", 25),
            temperature=self.config.temperature,
            max_output_tokens=getattr(self.config, "max_tokens", 512),
            prompt_builder=self.prompt_builder
        )

    def ingest_documents(self, file_paths: List[str]) -> int:
        """
        Nạp tài liệu từ danh sách file, cắt thành chunks, tạo embedding và lưu vào Vector Store.
        
        Args:
            file_paths: Danh sách đường dẫn file cần nạp.
            
        Returns:
            int: Tổng số lượng chunk đã được đánh chỉ mục.
            
        TODO:
            - Dùng self.loader.load_file cho từng đường dẫn để nhận Document.
            - Dùng self.chunker.chunk_documents để cắt các Document thành danh sách Chunk.
            - Dùng self.embedder.embed_chunks để tạo ma trận vector cho các Chunk.
            - Dùng self.vector_store.add_chunks để lưu vào kho.
            - Trả về số lượng chunk đã nạp.
        """
        # 1. Đọc từng file văn bản thành đối tượng Document
        documents = []
        for path in file_paths:
            documents.append(self.loader.load_file(path))
        if not documents:
            return 0
        # 2. Cắt các Document thành danh sách Chunk
        chunks = self.chunker.chunk_documents(documents)
        if not chunks:
            return 0
        # 3. Tạo vector embedding cho các Chunk bằng embedder
        embeddings = self.embedder.embed_chunks(chunks)
        # 4. Lưu vào kho vector (tự động commit vào SQLite ngoài đĩa chống crash)
        self.vector_store.add_chunks(chunks, embeddings)
        return len(chunks)

        

    def answer(self, query: str) -> GenerationOutput:
        """
        Nhận câu hỏi của người dùng và trả về câu trả lời hoàn chỉnh (kèm các context đã dùng).
        
        Args:
            query: Câu hỏi của người dùng.
            
        Returns:
            GenerationOutput: Câu trả lời của LLM và danh sách các chunk được trích xuất.
            
        TODO:
            - Gọi self.retriever.retrieve(query) để lấy Top-K SearchResult.
            - Gọi self.generator.generate(query, search_results).
            - Trả về kết quả sinh ra.
        """
        # 1. Gọi Retriever để tìm Top-K SearchResult
        search_results = self.retriever.retrieve(query)
        # 2. Gọi Generator để sinh câu trả lời dựa trên ngữ cảnh đã truy xuất
        generation_output = self.generator.generate(query, search_results)
        return generation_output

if __name__ == "__main__":
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    # Ví dụ về cách khởi chạy Simple RAG sau khi bạn đã hoàn thiện code bên trong các class:
    print("=== Khởi tạo Simple RAG Pipeline ===")
    config = SimpleRAGConfig()
    pipeline = SimpleRAGPipeline(config)
    print("Pipeline đã sẵn sàng. Hãy hoàn thiện nội dung các method để chạy thực tế!")
