from typing import List, Optional, Dict, Any
from schema import SearchResult, ClaimVerification, GroundednessReport
from config import SimpleRAGConfig
from evaluation.base_llm_judge import BaseLLMJudge

class GroundednessJudge(BaseLLMJudge):
    """
    Bộ chấm điểm Groundedness / Faithfulness (Tính trung thực, có căn cứ) bằng LLM-as-a-Judge.
    
    * ĐỊNH NGHĨA CHỈ SỐ:
      Groundedness (hay Faithfulness) đo lường mức độ câu trả lời của mô hình dựa HOÀN TOÀN
      trên ngữ cảnh được cung cấp (Retrieved Contexts), không tự bịa đặt hay ảo giác (Hallucination).
      - 1.0: Mọi mệnh đề trong câu trả lời đều có trích dẫn/bằng chứng rõ ràng trong tài liệu.
      - 0.0: Câu trả lời hoàn toàn là ảo giác hoặc mâu thuẫn với ngữ cảnh.
      
    * THUẬT TOÁN ATOMIC CLAIM VERIFICATION (Chuẩn theo Custom-RAG & bài báo ASTUTE RAG):
      1. Tách câu trả lời thành danh sách các mệnh đề nguyên tử (Atomic Claims).
         Ví dụ: "Hà Nội là thủ đô của Việt Nam và có diện tích hơn 3000 km2"
         -> Claim 1: "Hà Nội là thủ đô của Việt Nam"
         -> Claim 2: "Hà Nội có diện tích hơn 3000 km2"
      2. Đối chiếu từng Claim với các <chunk> trong ngữ cảnh:
         - supported: True nếu ngữ cảnh trực tiếp khẳng định điều này.
         - evidence_id: Ghi rõ chunk_id chứa bằng chứng.
      3. Điểm Groundedness = Số claim được chứng minh (supported) / Tổng số claim.
    """

    SYSTEM_PROMPT = (
        "Bạn là một giám khảo AI khắt khe chuyên kiểm định tính trung thực (Groundedness / Faithfulness) của câu trả lời.\n"
        "Nhiệm vụ:\n"
        "1. Phân rã <answer> thành các mệnh đề nguyên tử độc lập (Atomic Claims).\n"
        "2. Kiểm tra từng mệnh đề xem có được hỗ trợ trực tiếp bởi bất kỳ <chunk> nào trong <context> hay không.\n"
        "   - Đặt 'supported': true nếu có bằng chứng rõ ràng trong ngữ cảnh, kèm 'evidence_id' là chunk_id tương ứng.\n"
        "   - Đặt 'supported': false nếu mệnh đề không tìm thấy trong ngữ cảnh (bị ảo giác hoặc LLM tự suy đoán).\n"
        "3. Tính 'groundedness_score' = (số claim supported == true) / (tổng số claims).\n"
        "Yêu cầu trả về đúng định dạng JSON:\n"
        "{\n"
        "  \"groundedness_score\": 1.0,\n"
        "  \"is_grounded\": true,\n"
        "  \"reasoning\": \"Giải thích ngắn gọn bằng tiếng Việt\",\n"
        "  \"claims\": [\n"
        "    {\n"
        "      \"claim\": \"Nội dung mệnh đề nguyên tử\",\n"
        "      \"supported\": true,\n"
        "      \"evidence_id\": \"chunk_id\",\n"
        "      \"reason\": \"Trích đoạn làm bằng chứng\"\n"
        "    }\n"
        "  ]\n"
        "}"
    )

    def evaluate(self, answer: str, retrieved_contexts: List[SearchResult]) -> GroundednessReport:
        """
        Đánh giá tính trung thực và căn cứ của câu trả lời dựa trên ngữ cảnh đã truy xuất.
        
        Args:
            answer: Câu trả lời do LLM sinh ra.
            retrieved_contexts: Danh sách các đoạn trích dẫn được dùng làm đầu vào cho LLM.
            
        Returns:
            GroundednessReport: Báo cáo chi tiết từng mệnh đề và điểm trung thực.
        """
        if not answer or not answer.strip():
            return GroundednessReport(
                score=0.0,
                claims=[],
                is_grounded=False,
                judge_reasoning="Câu trả lời rỗng."
            )

        if not retrieved_contexts:
            return GroundednessReport(
                score=0.0,
                claims=[ClaimVerification(claim=answer, supported=False, evidence_id="", reason="Không có ngữ cảnh nào được cấp.")],
                is_grounded=False,
                judge_reasoning="Không có ngữ cảnh đầu vào để đối chiếu."
            )

        # 1. Gọi API nếu có API Key hợp lệ
        if self.has_valid_api_key:
            ctx_blocks = []
            for res in retrieved_contexts:
                ctx_blocks.append(
                    f"<chunk id=\"{res.chunk.chunk_id}\">\n"
                    f"{res.chunk.content}\n"
                    f"</chunk>"
                )

            user_prompt = (
                f"<context>\n{chr(10).join(ctx_blocks)}\n</context>\n\n"
                f"<answer>\n{answer}\n</answer>\n\n"
                "Hãy kiểm định từng mệnh đề trong câu trả lời trên."
            )

            parsed = self._call_api(self.SYSTEM_PROMPT, user_prompt)
            if parsed and "claims" in parsed:
                raw_claims = parsed.get("claims", [])
                claim_objs = [
                    ClaimVerification(
                        claim=str(c.get("claim", "")),
                        supported=bool(c.get("supported", False)),
                        evidence_id=str(c.get("evidence_id", "")),
                        reason=str(c.get("reason", ""))
                    )
                    for c in raw_claims if isinstance(c, dict) and "claim" in c
                ]
                
                if claim_objs:
                    score = sum(1 for c in claim_objs if c.supported) / float(len(claim_objs))
                else:
                    score = float(parsed.get("groundedness_score", 0.0))
                    
                score = max(0.0, min(1.0, score))
                return GroundednessReport(
                    score=round(score, 4),
                    claims=claim_objs,
                    is_grounded=(score >= 0.7),
                    judge_reasoning=str(parsed.get("reasoning", ""))
                )

        # 2. Chế độ Fallback mô phỏng (Heuristic khi chưa có API Key)
        # Tách theo dấu chấm câu và kiểm tra xem câu có xuất hiện trong context không
        sentences = [s.strip() for s in answer.split(".") if len(s.strip()) > 5]
        if not sentences:
            sentences = [answer.strip()]

        full_context_text = " ".join(r.chunk.content.lower() for r in retrieved_contexts)
        claim_objs = []
        supported_count = 0

        for s in sentences:
            # Nếu 60% từ quan trọng trong câu có trong context -> coi là supported
            words = [w for w in s.lower().split() if len(w) > 3]
            match_count = sum(1 for w in words if w in full_context_text)
            is_sup = (match_count / len(words) >= 0.5) if words else False
            if is_sup:
                supported_count += 1
            claim_objs.append(ClaimVerification(
                claim=s,
                supported=is_sup,
                evidence_id=retrieved_contexts[0].chunk.chunk_id if is_sup else "",
                reason="Mô phỏng: Kiểm tra từ khóa xuất hiện trong ngữ cảnh."
            ))

        score = round(supported_count / len(sentences), 4) if sentences else 0.0
        return GroundednessReport(
            score=score,
            claims=claim_objs,
            is_grounded=(score >= 0.7),
            judge_reasoning="[Chế độ mô phỏng] Cung cấp GEMINI_API_KEY trong file .env hoặc config để chạy LLM Atomic Verification."
        )
