from typing import List
from schema import Chunk, SearchResult, SourceType

class NoiseAndConflictSimulator:
    """
    Module hỗ trợ thực nghiệm: Mô phỏng các dạng lỗi truy xuất thực tế (Imperfect Retrieval & Knowledge Conflicts).
    
    Theo bài báo "ASTUTE RAG: Overcoming Imperfect Retrieval Augmentation and Knowledge Conflicts" (ACL 2025):
    - Standard RAG sụp đổ hoặc suy giảm hiệu năng mạnh khi gặp các tình huống retrieval này.
    - Lớp này giúp bạn tạo ra các kịch bản kiểm thử có kiểm soát (Controlled Scenarios) để đo lường.
    """

    def __init__(self):
        """Khởi tạo simulator."""
        pass

    def simulate_clean_retrieval(self, query: str, true_context: str) -> List[SearchResult]:
        """
        Kịch bản 1: Truy xuất hoàn hảo (Clean Retrieval).
        Mọi đoạn văn bản được trả về đều đúng và chứa câu trả lời.
        Ở kịch bản này, Standard RAG hoạt động rất tốt.
        """
        chunk = Chunk(
            chunk_id="clean_c1",
            doc_id="doc_true",
            content=true_context,
            source_type=SourceType.EXTERNAL
        )
        return [SearchResult(chunk=chunk, score=0.95, rank=1)]

    def simulate_imperfect_noisy_retrieval(
        self,
        query: str,
        true_context: str,
        noisy_contexts: List[str],
        noise_ratio: float = 0.7
    ) -> List[SearchResult]:
        """
        Kịch bản 2: Truy xuất thực tế có nhiễu (Imperfect Retrieval với 70% Noise).
        Phản ánh đúng thống kê ở Hình 2 bài báo ASTUTE RAG: ~70% đoạn trích từ công cụ tìm kiếm
        không chứa trực tiếp câu trả lời mà chứa thông tin rác, lạc đề.
        
        Args:
            query: Câu hỏi người dùng.
            true_context: Đoạn văn bản chứa đáp án đúng.
            noisy_contexts: Danh sách các đoạn văn bản lạc đề, không liên quan.
            noise_ratio: Tỷ lệ nhiễu (mặc định 0.7 = 70%).
            
        Returns:
            List[SearchResult]: Danh sách kết quả bị pha trộn giữa văn bản đúng và văn bản rác.
        """
        results = [
            SearchResult(
                chunk=Chunk(
                    chunk_id="true_c1",
                    doc_id="doc_true",
                    content=true_context,
                    source_type=SourceType.EXTERNAL
                ),
                score=0.92,
                rank=1
            )
        ]
        for i, noise in enumerate(noisy_contexts, start=2):
            results.append(
                SearchResult(
                    chunk=Chunk(
                        chunk_id=f"noisy_c{i}",
                        doc_id=f"doc_noisy_{i}",
                        content=noise,
                        source_type=SourceType.EXTERNAL
                    ),
                    score=max(0.1, round(0.92 - (i - 1) * 0.08, 2)),
                    rank=i
                )
            )
        return results

    def simulate_misleading_conflict(
        self,
        query: str,
        ground_truth: str,
        counterfactual_context: str
    ) -> List[SearchResult]:
        """
        Kịch bản 3: Xung đột tri thức với thông tin giả mạo/sai lệch (Misleading Knowledge Conflict).
        Ví dụ trong bài báo (Hình 1 & 12):
        - Câu hỏi: "Ai đóng vai David trong phim Alvin and the Chipmunks?"
        - LLM nhớ sẵn trong đầu: "Jason Lee".
        - Đoạn văn bản bên ngoài lại ghi: "Zachary Levi đóng vai David..." (thông tin sai/nhầm nhân vật).
        
        -> Standard RAG sẽ bị "lừa" và trả lời sai thành "Zachary Levi", trong khi No-RAG lại đúng!
        """
        chunk = Chunk(
            chunk_id="misleading_c1",
            doc_id="doc_counterfactual",
            content=counterfactual_context,
            source_type=SourceType.EXTERNAL
        )
        return [SearchResult(chunk=chunk, score=0.96, rank=1)]

    def simulate_worst_case_rgb(self, query: str, negative_contexts: List[str]) -> List[SearchResult]:
        """
        Kịch bản 4: Kịch bản tồi tệ nhất (Worst-case Scenario theo benchmark RGB - Hình 6 bài báo).
        Toàn bộ 100% các đoạn truy xuất đều là tiêu cực (negative/counterfactual).
        - Standard RAG giảm hơn 50% độ chính xác so với việc không dùng RAG (No-RAG).
        - ASTUTE RAG là phương pháp duy nhất duy trì độ chính xác tương đương hoặc vượt qua No-RAG.
        """
        results = []
        for i, neg in enumerate(negative_contexts, start=1):
            results.append(
                SearchResult(
                    chunk=Chunk(
                        chunk_id=f"negative_c{i}",
                        doc_id=f"doc_neg_{i}",
                        content=neg,
                        source_type=SourceType.EXTERNAL
                    ),
                    score=max(0.1, round(0.95 - (i - 1) * 0.05, 2)),
                    rank=i
                )
            )
        return results
