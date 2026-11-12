from typing import List, Optional, Dict, Any
from schema import SearchResult, ContextRelevanceReport
from config import SimpleRAGConfig
from evaluation.base_llm_judge import BaseLLMJudge

class ContextRelevanceJudge(BaseLLMJudge):
    """
    Bộ chấm điểm Context Relevance (Độ liên quan của ngữ cảnh truy xuất) bằng LLM-as-a-Judge.
    
    * ĐỊNH NGHĨA CHỈ SỐ:
      Context Relevance đo lường tỷ lệ thông tin trong các đoạn văn bản truy xuất (Retrieved Contexts)
      thực sự liên quan và hữu ích để trả lời câu hỏi của người dùng.
      - 1.0: Tất cả các đoạn đều tập trung, liên quan trực tiếp đến câu hỏi.
      - 0.5: Có đoạn liên quan, nhưng bị pha lẫn các đoạn lạc đề hoặc nhiễu rác.
      - 0.0: Toàn bộ các đoạn truy xuất đều không liên quan đến câu hỏi.
      
    * THUẬT TOÁN (Tương tự Custom-RAG & Ragas):
      1. Gửi câu hỏi và danh sách các đoạn trích tới LLM.
      2. Yêu cầu LLM đánh giá từng đoạn:
         - is_relevant: true/false.
         - relevance_score: [0.0 -> 1.0].
         - key_sentences: Các câu thực sự chứa thông tin trả lời.
         - reason: Lý do giải thích.
      3. Điểm tổng hợp = Trung bình cộng relevance_score của các đoạn.
    """

    SYSTEM_PROMPT = (
        "Bạn là một giám khảo AI công tâm chuyên đánh giá hệ thống RAG (Retrieval-Augmented Generation).\n"
        "Nhiệm vụ của bạn là đánh giá chỉ số CONTEXT RELEVANCE (Độ liên quan của ngữ cảnh truy xuất):\n"
        "Hãy kiểm tra xem từng đoạn ngữ cảnh (<chunk>) có chứa thông tin trực tiếp hoặc gián tiếp hữu ích "
        "để trả lời cho <query> hay không.\n"
        "Yêu cầu trả về đúng định dạng JSON:\n"
        "{\n"
        "  \"overall_score\": 0.85,\n"
        "  \"reasoning\": \"Tóm tắt lý do tổng thể bằng tiếng Việt\",\n"
        "  \"evaluated_chunks\": [\n"
        "    {\n"
        "      \"chunk_id\": \"mã chunk\",\n"
        "      \"is_relevant\": true,\n"
        "      \"relevance_score\": 1.0,\n"
        "      \"reason\": \"Giải thích ngắn gọn\"\n"
        "    }\n"
        "  ]\n"
        "}"
    )

    def evaluate(self, query: str, retrieved_contexts: List[SearchResult]) -> ContextRelevanceReport:
        """
        Đánh giá độ liên quan của danh sách SearchResult đối với câu hỏi query.
        
        Args:
            query: Câu hỏi của người dùng.
            retrieved_contexts: Danh sách SearchResult do Retriever trả về.
            
        Returns:
            ContextRelevanceReport: Báo cáo chi tiết điểm số và đánh giá từng chunk.
        """
        if not retrieved_contexts:
            return ContextRelevanceReport(
                score=0.0,
                relevant_chunks_count=0,
                total_chunks_count=0,
                chunk_scores=[],
                judge_reasoning="Không có ngữ cảnh nào được truy xuất."
            )

        # 1. Gọi API nếu có API Key hợp lệ
        if self.has_valid_api_key:
            chunk_blocks = []
            for i, res in enumerate(retrieved_contexts, start=1):
                chunk_blocks.append(
                    f"<chunk id=\"{res.chunk.chunk_id}\" rank=\"{res.rank}\">\n"
                    f"{res.chunk.content}\n"
                    f"</chunk>"
                )
            
            user_prompt = (
                f"<query>{query}</query>\n\n"
                f"<retrieved_chunks>\n"
                f"{chr(10).join(chunk_blocks)}\n"
                f"</retrieved_chunks>\n\n"
                "Hãy chấm điểm độ liên quan của từng chunk và toàn bộ ngữ cảnh đối với câu hỏi trên."
            )

            parsed = self._call_api(self.SYSTEM_PROMPT, user_prompt)
            if parsed and "overall_score" in parsed:
                eval_chunks = parsed.get("evaluated_chunks", [])
                score = max(0.0, min(1.0, float(parsed.get("overall_score", 0.0))))
                rel_count = sum(1 for c in eval_chunks if c.get("is_relevant", False))
                return ContextRelevanceReport(
                    score=round(score, 4),
                    relevant_chunks_count=rel_count,
                    total_chunks_count=len(retrieved_contexts),
                    chunk_scores=eval_chunks,
                    judge_reasoning=str(parsed.get("reasoning", ""))
                )

        # 2. Chế độ Fallback mô phỏng (Heuristic khi chưa có API Key)
        # Tính điểm dựa trên tỷ lệ từ khóa câu hỏi xuất hiện trong chunk
        query_words = set(query.lower().split())
        chunk_scores = []
        rel_count = 0
        total_score = 0.0

        for res in retrieved_contexts:
            content_words = set(res.chunk.content.lower().split())
            common = query_words & content_words
            ratio = len(common) / len(query_words) if query_words else 0.0
            is_rel = ratio > 0.2
            if is_rel:
                rel_count += 1
            s = min(1.0, ratio * 2.0)
            total_score += s
            chunk_scores.append({
                "chunk_id": res.chunk.chunk_id,
                "is_relevant": is_rel,
                "relevance_score": round(s, 2),
                "reason": f"Mô phỏng (chưa cấu hình API Key): Trùng khớp {len(common)} từ khóa."
            })

        avg_score = round(total_score / len(retrieved_contexts), 4) if retrieved_contexts else 0.0
        return ContextRelevanceReport(
            score=avg_score,
            relevant_chunks_count=rel_count,
            total_chunks_count=len(retrieved_contexts),
            chunk_scores=chunk_scores,
            judge_reasoning="[Chế độ mô phỏng] Cung cấp GEMINI_API_KEY trong file .env hoặc config để chấm bằng LLM thật."
        )
