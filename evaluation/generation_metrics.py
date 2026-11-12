import re
import string
from collections import Counter
from typing import Dict, List, Tuple
from schema import GenerationOutput, EvalSample

class SimpleGenerationEvaluator:
    """
    Giai đoạn 6B: Đánh giá chất lượng của Tầng Sinh câu trả lời (Generation Evaluation)
    và Phân tích Mâu thuẫn Tri thức (Knowledge Conflicts Analysis theo bài báo ASTUTE RAG).
    
    * CÁC CHỈ SỐ CƠ BẢN:
      - Exact Match (EM): Câu trả lời của mô hình có khớp chính xác với đáp án chuẩn hay không?
      - F1-Score: Độ trùng lặp token giữa câu trả lời và đáp án chuẩn.
      
    * PHÂN TÍCH XUNG ĐỘT TRI THỨC (Theo Mục 3 & Mục 5.3 của ASTUTE RAG):
      Phân loại kết quả của từng câu hỏi thành 3 nhóm kịch bản:
      1. Cả 2 cùng đúng (Both Correct): LLM đúng và RAG cũng đúng.
      2. Cả 2 cùng sai (Both Incorrect): Cả LLM lẫn RAG đều không trả lời được.
      3. Xung đột tri thức (Conflicting): Một bên đúng và một bên sai.
         Đặc biệt xem xét trường hợp: LLM tự trả lời đúng, nhưng khi thêm RAG (với tài liệu nhiễu)
         thì RAG lại trả lời SAI -> Đây chính là "The Pitfall of RAG"!
    """

    def __init__(self):
        """Khởi tạo evaluator."""
        pass

    @staticmethod
    def normalize_text(text: str) -> str:
        """
        Chuẩn hóa chuỗi văn bản trước khi chấm điểm:
        - Chuyển thành chữ thường.
        - Xóa dấu câu, mạo từ (a, an, the trong tiếng Anh), khoảng trắng thừa.
        
        Args:
            text: Chuỗi văn bản cần chuẩn hóa.
            
        Returns:
            str: Chuỗi văn bản sạch.
        """
        text = text.lower()
        text = "".join(ch for ch in text if ch not in string.punctuation)
        words = [w for w in text.split() if w not in {"a", "an", "the"}]
        return " ".join(words).strip()

    def compute_exact_match(self, prediction: str, ground_truth: str) -> float:
        """
        Tính điểm Exact Match (EM): Trả về 1.0 nếu chuỗi chuẩn hóa trùng khớp, ngược lại 0.0.
        
        Args:
            prediction: Câu trả lời do mô hình sinh ra.
            ground_truth: Đáp án chuẩn (nhãn).
            
        TODO:
            - Chuẩn hóa prediction và ground_truth bằng self.normalize_text.
            - So sánh bằng (==) và trả về 1.0 hoặc 0.0.
        """
        return 1.0 if self.normalize_text(prediction) == self.normalize_text(ground_truth) else 0.0

    def compute_f1(self, prediction: str, ground_truth: str) -> float:
        """
        Tính điểm F1-Score theo mức độ trùng lặp tập token (từ).
        
        Args:
            prediction: Câu trả lời mô hình.
            ground_truth: Đáp án chuẩn.
            
        Returns:
            float: Điểm F1 trong khoảng [0.0, 1.0].
            
        TODO:
            - Tách token của prediction và ground_truth.
            - Tìm giao của 2 tập token (dùng Counter đa tập).
            - Tính Precision, Recall và F1 = 2 * (P * R) / (P + R).
        """
        pred_tokens = self.normalize_text(prediction).split()
        truth_tokens = self.normalize_text(ground_truth).split()

        if not pred_tokens or not truth_tokens:
            return 0.0

        common = Counter(pred_tokens) & Counter(truth_tokens)
        num_common = sum(common.values())

        if num_common == 0:
            return 0.0

        precision = num_common / len(pred_tokens)
        recall = num_common / len(truth_tokens)
        return round(2 * (precision * recall) / (precision + recall), 4)

    def analyze_knowledge_conflict(
        self,
        no_rag_output: GenerationOutput,
        rag_output: GenerationOutput,
        ground_truth: str
    ) -> Dict[str, Any]:
        """
        Phân loại tương quan tri thức giữa LLM nội tại và RAG bên ngoài theo bài báo ASTUTE RAG (Hình 1 & 8).
        
        Returns:
            Dict: Chứa nhãn phân loại:
                  - 'category': "BOTH_CORRECT" | "BOTH_INCORRECT" | "LLM_CORRECT_RAG_WRONG" | "RAG_CORRECT_LLM_WRONG"
                  - 'is_conflict': True nếu 1 bên đúng 1 bên sai.
                  - 'rag_degradation': True nếu LLM đúng nhưng thêm RAG vào lại bị SAI (Tác hại của Imperfect Retrieval).
        """
        llm_correct = (self.compute_exact_match(no_rag_output.answer, ground_truth) == 1.0) or (self.compute_f1(no_rag_output.answer, ground_truth) >= 0.5)
        rag_correct = (self.compute_exact_match(rag_output.answer, ground_truth) == 1.0) or (self.compute_f1(rag_output.answer, ground_truth) >= 0.5)

        if llm_correct and rag_correct:
            category = "BOTH_CORRECT"
        elif not llm_correct and not rag_correct:
            category = "BOTH_INCORRECT"
        elif llm_correct and not rag_correct:
            category = "LLM_CORRECT_RAG_WRONG"  # The Pitfall of RAG!
        else:
            category = "RAG_CORRECT_LLM_WRONG"

        return {
            "category": category,
            "is_conflict": (llm_correct != rag_correct),
            "rag_degradation": (llm_correct and not rag_correct),
            "llm_correct": llm_correct,
            "rag_correct": rag_correct,
            "llm_em": self.compute_exact_match(no_rag_output.answer, ground_truth),
            "rag_em": self.compute_exact_match(rag_output.answer, ground_truth),
            "llm_f1": self.compute_f1(no_rag_output.answer, ground_truth),
            "rag_f1": self.compute_f1(rag_output.answer, ground_truth)
        }
