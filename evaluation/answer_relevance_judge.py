from typing import Optional, Dict, Any
from schema import AnswerRelevanceReport
from config import SimpleRAGConfig
from evaluation.base_llm_judge import BaseLLMJudge

class AnswerRelevanceJudge(BaseLLMJudge):
    """
    Bộ chấm điểm Answer Relevance (Mức độ phù hợp và thỏa đáng của câu trả lời) bằng LLM-as-a-Judge.
    
    * ĐỊNH NGHĨA CHỈ SỐ:
      Answer Relevance đo lường xem câu trả lời do LLM sinh ra có trực tiếp giải quyết trọng tâm
      của câu hỏi hay không:
      - 1.0: Trực tiếp, súc tích, giải quyết trọn vẹn thắc mắc của người dùng.
      - 0.5: Trả lời vòng vo, chung chung, hoặc chỉ trả lời được một phần câu hỏi.
      - 0.0: Lạc đề hoàn toàn, thoái thác không có lý do, hoặc câu trả lời vô nghĩa.
      
    * THUẬT TOÁN:
      1. Gửi cặp (Câu hỏi - Câu trả lời) tới LLM Judge.
      2. LLM phân tích mức độ tương thích ngữ nghĩa (Semantic Alignment) và tính đầy đủ (Completeness).
      3. Trả về điểm số trong đoạn [0.0 -> 1.0] kèm nhận xét chi tiết.
    """

    SYSTEM_PROMPT = (
        "Bạn là một giám khảo AI chuyên đánh giá chất lượng câu trả lời trong hệ thống Hỏi-Đáp (QA).\n"
        "Nhiệm vụ của bạn là chấm điểm ANSWER RELEVANCE (Mức độ phù hợp của câu trả lời với câu hỏi):\n"
        "Quy tắc chấm điểm:\n"
        "- 1.0: Câu trả lời đi thẳng vào vấn đề, giải quyết đúng và đủ yêu cầu của câu hỏi.\n"
        "- 0.7 - 0.9: Trả lời đúng trọng tâm nhưng hơi dài dòng hoặc thừa thông tin phụ.\n"
        "- 0.4 - 0.6: Chỉ trả lời được một phần câu hỏi, hoặc có ý đúng nhưng chưa trọn vẹn.\n"
        "- 0.1 - 0.3: Trả lời mơ hồ, vòng vo, tránh né câu hỏi.\n"
        "- 0.0: Hoàn toàn lạc đề, vô nghĩa hoặc không liên quan đến câu hỏi.\n\n"
        "Yêu cầu trả về đúng định dạng JSON:\n"
        "{\n"
        "  \"relevance_score\": 0.9,\n"
        "  \"is_relevant\": true,\n"
        "  \"reasoning\": \"Phân tích ngắn gọn lý do cho điểm bằng tiếng Việt\"\n"
        "}"
    )

    def evaluate(self, query: str, answer: str, ground_truth: Optional[str] = None) -> AnswerRelevanceReport:
        """
        Đánh giá độ phù hợp của câu trả lời đối với câu hỏi query.
        
        Args:
            query: Câu hỏi ban đầu của người dùng.
            answer: Câu trả lời do hệ thống RAG sinh ra.
            ground_truth: Đáp án chuẩn mẫu (tùy chọn, dùng để đối chiếu thêm nếu có).
            
        Returns:
            AnswerRelevanceReport: Báo cáo điểm số và phân tích của giám khảo.
        """
        if not answer or not answer.strip():
            return AnswerRelevanceReport(
                score=0.0,
                is_relevant=False,
                judge_reasoning="Câu trả lời rỗng."
            )

        # 1. Gọi API nếu có API Key hợp lệ
        if self.has_valid_api_key:
            ref_part = f"\n<ground_truth_reference>{ground_truth}</ground_truth_reference>" if ground_truth else ""
            user_prompt = (
                f"<query>{query}</query>\n"
                f"<answer>{answer}</answer>"
                f"{ref_part}\n\n"
                "Hãy đánh giá mức độ phù hợp và thỏa đáng của câu trả lời trên đối với câu hỏi."
            )

            parsed = self._call_api(self.SYSTEM_PROMPT, user_prompt)
            if parsed and "relevance_score" in parsed:
                score = max(0.0, min(1.0, float(parsed.get("relevance_score", 0.0))))
                is_rel = bool(parsed.get("is_relevant", score >= 0.7))
                return AnswerRelevanceReport(
                    score=round(score, 4),
                    is_relevant=is_rel,
                    judge_reasoning=str(parsed.get("reasoning", ""))
                )

        # 2. Chế độ Fallback mô phỏng (Heuristic khi chưa có API Key)
        # So sánh độ giao thoa từ vựng giữa query và answer
        q_words = set(query.lower().split())
        a_words = set(answer.lower().split())
        overlap = len(q_words & a_words) / len(q_words) if q_words else 0.0

        # Nếu có ground truth, kiểm tra thêm độ trùng với ground truth
        gt_score = 0.0
        if ground_truth:
            gt_words = set(ground_truth.lower().split())
            if gt_words:
                gt_score = len(gt_words & a_words) / len(gt_words)

        final_score = round(min(1.0, max(0.3, overlap * 1.5 + gt_score * 0.5)), 2)
        return AnswerRelevanceReport(
            score=final_score,
            is_relevant=(final_score >= 0.7),
            judge_reasoning="[Chế độ mô phỏng] Cung cấp GEMINI_API_KEY trong file .env hoặc config để chấm điểm Answer Relevance bằng LLM."
        )
