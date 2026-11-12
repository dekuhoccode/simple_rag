from typing import List, Dict, Any
from schema import SearchResult, EvalSample

class SimpleRetrievalEvaluator:
    """
    Giai đoạn 6A: Đánh giá chất lượng của Tầng Truy xuất (Retrieval Evaluation).
    
    Đo lường mức độ hiệu quả của Retriever bằng các chỉ số Information Retrieval (IR) chuẩn:
    - Hit Rate@K: Có tìm thấy ít nhất 1 thông tin mục tiêu trong Top-K không?
    - Recall@K: Tìm được bao nhiêu phần trăm thông tin mục tiêu?
    - MRR@K (Mean Reciprocal Rank): Thứ hạng của kết quả đúng đầu tiên là bao nhiêu?
    """

    def __init__(self):
        """Khởi tạo evaluator."""
        pass

    def evaluate_sample(self, sample: EvalSample, retrieved_results: List[SearchResult], k: int = 3) -> Dict[str, float]:
        """
        Tính điểm các chỉ số truy xuất cho một mẫu kiểm thử đơn lẻ.
        
        Args:
            sample: Mẫu kiểm thử chứa danh sách target_keywords (từ khóa đáp án bắt buộc có).
            retrieved_results: Danh sách các đoạn do Retriever tìm được.
            k: Ngưỡng cắt Top-K.
            
        Returns:
            Dict[str, float]: Từ điển chứa {'hit_rate': 0.0 hoặc 1.0, 'mrr': float, 'recall': float}.
            
        TODO:
            - Lấy Top-K kết quả đầu tiên.
            - Kiểm tra xem target_keywords có xuất hiện trong nội dung chunk.content hay không.
            - Tính Hit Rate (1.0 nếu có ít nhất 1 từ khóa trúng, ngược lại 0.0).
            - Tính MRR (1 / vị trí rank của đoạn đúng đầu tiên).
            - Tính Recall (số từ khóa trúng / tổng số từ khóa cần tìm).
        """
        top_results = retrieved_results[:k]
        keywords = sample.target_keywords
        if not keywords:
            return {"hit_rate": 0.0, "mrr": 0.0, "recall": 0.0}

        first_hit_rank = None
        for res in top_results:
            content_lower = res.chunk.content.lower()
            if any(kw.lower() in content_lower for kw in keywords):
                first_hit_rank = res.rank
                break

        hit_rate = 1.0 if first_hit_rank is not None else 0.0

        mrr = (1.0 / first_hit_rank) if first_hit_rank is not None else 0.0

        all_top_text = " ".join(r.chunk.content.lower() for r in top_results)
        matched_count = sum(1 for kw in keywords if kw.lower() in all_top_text)
        recall = matched_count / float(len(keywords))

        return {
            "hit_rate": round(hit_rate, 4),
            "mrr": round(mrr, 4),
            "recall": round(recall, 4)
        }

    def evaluate_dataset(self, samples: List[EvalSample], results_list: List[List[SearchResult]], k: int = 3) -> Dict[str, float]:
        """
        Tính trung bình cộng các chỉ số trên toàn bộ tập dữ liệu kiểm thử.
        
        Args:
            samples: Danh sách tất cả các câu hỏi kiểm thử.
            results_list: Danh sách các kết quả truy xuất tương ứng với từng câu hỏi.
            k: Ngưỡng Top-K.
            
        Returns:
            Dict[str, float]: Điểm trung bình của Mean Hit Rate, Mean MRR trên toàn tập.
            
        TODO:
            - Duyệt qua từng cặp (sample, results), gọi evaluate_sample.
            - Tính giá trị trung bình (average) cho từng chỉ số.
        """
        if not samples or len(samples) != len(results_list):
            return {"mean_hit_rate": 0.0, "mean_mrr": 0.0, "mean_recall": 0.0}

        all_hit = []
        all_mrr = []
        all_recall = []

        for sample, results in zip(samples, results_list):
            metrics = self.evaluate_sample(sample, results, k=k)
            all_hit.append(metrics["hit_rate"])
            all_mrr.append(metrics["mrr"])
            all_recall.append(metrics["recall"])

        n = len(samples)
        return {
            "mean_hit_rate": round(sum(all_hit) / n, 4),
            "mean_mrr": round(sum(all_mrr) / n, 4),
            "mean_recall": round(sum(all_recall) / n, 4)
        }
