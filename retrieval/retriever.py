from typing import List
from schema import SearchResult
from indexing.embedder import SimpleEmbedder
from indexing.vector_store import SimpleVectorStore

class SimpleRetriever:
    """
    Giai đoạn 4: Retrieval (Truy xuất thông tin).
    
    Trong kiến trúc Standard / Naive RAG:
    - Nhận câu hỏi tự nhiên của người dùng.
    - Gọi Embedder để tạo query vector.
    - So khớp với Vector Store và lấy Top-K đoạn văn bản có điểm Cosine cao nhất.
    
    * LƯU Ý HỌC TẬP & HẠN CHẾ SO VỚI BÀI BÁO ASTUTE RAG:
      1. SimpleRetriever "mù quáng" tin rằng mọi tài liệu có điểm Cosine cao đều chứa câu trả lời đúng.
      2. Trong thực tế (như thống kê trong bài báo ASTUTE RAG, Hình 2):
         - Khoảng 70% các đoạn truy xuất từ Search Engine/Corpus là "Imperfect" (chứa thông tin nhiễu,
           lạc đề, hoặc thậm chí sai lệch, mâu thuẫn).
      3. SimpleRetriever không có cơ chế:
         - Đánh giá độ tin cậy của nguồn (Source Reliability).
         - Phát hiện xung đột giữa các đoạn truy xuất với nhau.
         - Phát hiện xung đột giữa đoạn truy xuất với tri thức sẵn có của LLM.
    """

    def __init__(self, embedder: SimpleEmbedder, vector_store: SimpleVectorStore, top_k: int = 3):
        """
        Khởi tạo Retriever liên kết với bộ nhúng và kho vector.
        
        Args:
            embedder: Thực thể SimpleEmbedder dùng để nhúng câu truy vấn.
            vector_store: Kho SimpleVectorStore chứa các chunk đã được đánh chỉ mục.
            top_k: Số lượng chunk tối đa cần lấy ra cho mỗi câu hỏi.
        """
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(self, query: str) -> List[SearchResult]:
        """
        Thực hiện toàn bộ quy trình truy xuất cho một câu hỏi.
        
        Args:
            query: Câu hỏi của người dùng (dạng chuỗi văn bản tự nhiên).
            
        Returns:
            List[SearchResult]: Danh sách Top-K đoạn văn bản được xếp hạng theo độ tương đồng giảm dần.
            
        TODO:
            - Kiểm tra chuỗi query không rỗng.
            - Gọi self.embedder.embed_query(query) để lấy query_vector.
            - Gọi self.vector_store.similarity_search(query_vector, top_k=self.top_k).
            - Trả về danh sách kết quả.
        """
        if not query or not query.strip():
            return []
        q_vec = self.embedder.embed_query(query)
        results = self.vector_store.similarity_search(q_vec, top_k=self.top_k)
        return results
