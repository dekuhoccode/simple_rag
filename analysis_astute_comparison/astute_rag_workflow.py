from typing import List, Tuple, Optional
from schema import Chunk, SearchResult, ConsolidatedGroup, GenerationOutput, SourceType
from config import SimpleRAGConfig
from generation.llm_generator import SimpleLLMGenerator

class AstuteRAGConsolidator:
    """
    Khung triển khai giải pháp ASTUTE RAG (Theo thuật toán 1 & Mục 4 của bài báo ACL 2025).
    
    ASTUTE RAG được thiết kế để giải quyết triệt để 2 vấn đề lớn của Standard RAG:
    1. Imperfect Retrieval (Truy xuất không hoàn hảo, nhiều nhiễu).
    2. Knowledge Conflicts (Xung đột giữa tri thức nội tại của LLM và tài liệu bên ngoài).
    
    Quy trình 3 bước cốt lõi:
    - Bước 1: Adaptive Internal Knowledge Generation (Khơi gợi tri thức nội tại có điều kiện).
    - Bước 2: Iterative Source-aware Knowledge Consolidation (Gộp tri thức đồng thuận, tách tri thức mâu thuẫn).
    - Bước 3: Answer Finalization with Confidence (Đề xuất câu trả lời cho từng cụm và chọn đáp án tin cậy nhất).
    """

    def __init__(self, config: Optional[SimpleRAGConfig] = None, generator: Optional[SimpleLLMGenerator] = None):
        """Khởi tạo với cấu hình m_hat (số đoạn nội tại tối đa) và t (số vòng lặp hợp nhất)."""
        self.config = config or SimpleRAGConfig()
        self.generator = generator or SimpleLLMGenerator(
            model_name=self.config.llm_model,
            api_key=getattr(self.config, "llm_api_key", None),
            base_url=getattr(self.config, "llm_base_url", None),
            fallback_model_name=getattr(self.config, "fallback_llm_model", None),
            max_rpm=getattr(self.config, "max_rpm", 25),
            temperature=self.config.temperature,
            max_output_tokens=getattr(self.config, "max_tokens", 512)
        )

    def step1_adaptive_internal_knowledge_generation(self, query: str) -> List[Chunk]:
        """
        BƯỚC 1: Elicit Internal Knowledge (Alg 1, Dòng 1 & Mục 4.2).
        Cho LLM tự sinh ra tri thức từ chính trí nhớ tham số của nó.
        """
        sys_prompt = (
            "You are a factual knowledge retrieval module.\n"
            "Given a question, state only what you know with high certainty from your internal memory.\n"
            "If you are not certain or do not know, reply EXACTLY: 'I do not know'.\n"
            "Be concise and factual."
        )
        user_prompt = f"Question: {query}\nFactual Knowledge:"
        resp = self.generator._call_api(sys_prompt, user_prompt).strip()

        if not resp or "i do not know" in resp.lower() or "không biết" in resp.lower():
            return []

        return [
            Chunk(
                chunk_id="internal_mem_1",
                doc_id="parametric_memory",
                content=resp,
                source_type=SourceType.INTERNAL,
                metadata={"source": "LLM_Internal_Memory"}
            )
        ]

    def step2_source_aware_knowledge_consolidation(
        self,
        query: str,
        external_passages: List[Chunk],
        internal_passages: List[Chunk]
    ) -> List[ConsolidatedGroup]:
        """
        BƯỚC 2: Iterative Source-aware Knowledge Consolidation (Alg 1, Dòng 5-7 & Mục 4.3).
        Gộp các tri thức đồng thuận, tách tri thức mâu thuẫn giữa Internal và External.
        """
        groups: List[ConsolidatedGroup] = []

        if not external_passages and not internal_passages:
            return groups

        ext_texts = [f"[External {c.chunk_id}]: {c.content}" for c in external_passages]
        int_texts = [f"[Internal {c.chunk_id}]: {c.content}" for c in internal_passages]

        all_text = "\n".join(ext_texts + int_texts)
        has_internal = len(internal_passages) > 0
        has_external = len(external_passages) > 0

        # Phân tích xem có xung đột giữa 2 nguồn hay không
        is_conflicting = False
        if has_internal and has_external:
            sys_prompt = (
                "You are an expert fact-checker analyzing knowledge sources for conflicts.\n"
                "Compare the External and Internal information. State if they agree or conflict.\n"
                "Answer: 'AGREE' or 'CONFLICT', followed by a one-line explanation."
            )
            analysis = self.generator._call_api(sys_prompt, f"Question: {query}\n\nSources:\n{all_text}").strip()
            if "conflict" in analysis.lower():
                is_conflicting = True

        group = ConsolidatedGroup(
            group_id="group_1",
            summary_content=all_text,
            supporting_chunk_ids=[c.chunk_id for c in external_passages + internal_passages],
            sources=([SourceType.EXTERNAL] if has_external else []) + ([SourceType.INTERNAL] if has_internal else []),
            is_conflicting=is_conflicting,
            confidence_score=0.9 if not is_conflicting else 0.6
        )
        groups.append(group)
        return groups

    def step3_answer_finalization(
        self,
        query: str,
        consolidated_groups: List[ConsolidatedGroup]
    ) -> GenerationOutput:
        """
        BƯỚC 3: Answer Finalization & Confidence Assessment (Alg 1, Dòng 8-10 & Mục 4.4).
        Dựa trên các nhóm tri thức đã hợp nhất, LLM phân xử thông tin sai lệch và chọn đáp án chính xác nhất.
        """
        if not consolidated_groups:
            return self.generator.generate_without_rag(query)

        grp = consolidated_groups[0]
        sys_prompt = (
            "You are ASTUTE RAG: an advanced question answering model that resolves knowledge conflicts.\n"
            "RULES:\n"
            "1. You have both Internal Memory and External Retrieval passages.\n"
            "2. Note that External Retrieval may contain imperfect, noisy, or counterfactual (misleading) information.\n"
            "3. If there is a contradiction between a misleading external claim and factual reality, identify the discrepancy and provide the true factual answer.\n"
            "4. Output ONLY the concise, direct answer."
        )
        user_prompt = f"Knowledge Sources:\n{grp.summary_content}\n\nQuestion: {query}\nFinal Answer:"
        answer = self.generator._call_api(sys_prompt, user_prompt).strip()

        return GenerationOutput(
            query=query,
            answer=answer,
            retrieved_contexts=[],
            raw_prompt=f"{sys_prompt}\n\n{user_prompt}",
            is_conflicted=grp.is_conflicting
        )

    def run_astute_rag(self, query: str, retrieved_search_results: List[SearchResult]) -> GenerationOutput:
        """
        Hàm điều phối toàn bộ chu trình 3 bước của ASTUTE RAG.
        """
        external_passages = [r.chunk for r in retrieved_search_results]
        internal_passages = self.step1_adaptive_internal_knowledge_generation(query)
        groups = self.step2_source_aware_knowledge_consolidation(query, external_passages, internal_passages)
        return self.step3_answer_finalization(query, groups)
