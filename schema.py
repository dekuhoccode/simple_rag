from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from enum import Enum
import numpy as np

class SourceType(Enum):
    """
    Phân loại nguồn gốc của tri thức (Khái niệm cốt lõi của ASTUTE RAG: Source-awareness).
    - INTERNAL: Tri thức nằm sẵn trong trọng số tham số của LLM (LLM Parametric Memory).
    - EXTERNAL: Tri thức truy xuất từ tài liệu/web bên ngoài (Retrieved Passages).
    """
    INTERNAL = "internal_memory"
    EXTERNAL = "external_retrieval"


@dataclass
class Document:
    """
    Lớp đại diện cho một tài liệu thô ban đầu (file text, markdown, web page).
    """
    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Chunk:
    """
    Đơn vị văn bản nhỏ sau khi cắt từ Document, sẵn sàng để tạo Embedding và Indexing.
    """
    chunk_id: str
    doc_id: str
    content: str
    source_type: SourceType = SourceType.EXTERNAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    embedding: Optional[np.ndarray] = None


@dataclass
class SearchResult:
    """
    Kết quả trả về từ bộ truy xuất (Retriever).
    Bao gồm chunk tìm được và điểm tương đồng (similarity score).
    """
    chunk: Chunk
    score: float
    rank: int


@dataclass
class GenerationOutput:
    """
    Kết quả sinh ra từ mô hình ngôn ngữ lớn (LLM).
    """
    query: str
    answer: str
    retrieved_contexts: List[SearchResult] = field(default_factory=list)
    raw_prompt: str = ""
    is_conflicted: bool = False  # Đánh dấu nếu phát hiện xung đột tri thức trong ngữ cảnh


@dataclass
class ConsolidatedGroup:
    """
    Cấu trúc nhóm tri thức sau khi thực hiện Source-aware Consolidation theo ASTUTE RAG.
    Dùng để chỉ ra cách khắc phục nhược điểm của Standard RAG.
    """
    group_id: str
    summary_content: str
    supporting_chunk_ids: List[str] = field(default_factory=list)
    sources: List[SourceType] = field(default_factory=list)
    is_conflicting: bool = False
    proposed_candidate_answer: str = ""
    confidence_score: float = 0.0


@dataclass
class EvalSample:
    """
    Mẫu dữ liệu kiểm thử (Golden Dataset) để chạy đánh giá.
    """
    sample_id: str
    query: str
    ground_truth_answer: str
    target_keywords: List[str] = field(default_factory=list)
    is_noise_injected: bool = False  # Đánh dấu mẫu có bị tiêm nhiễu/xung đột để thử nghiệm


# =====================================================================
# CÁC BÁO CÁO ĐÁNH GIÁ RAG TRIAD (LLM-AS-A-JUDGE METRICS)
# =====================================================================

@dataclass
class ClaimVerification:
    """Kết quả kiểm chứng từng mệnh đề (claim) đơn lẻ trong câu trả lời của LLM."""
    claim: str
    supported: bool          # True nếu mệnh đề có căn cứ trực tiếp trong ngữ cảnh
    evidence_id: str = ""    # Mã chunk chứng minh mệnh đề đó
    reason: str = ""


@dataclass
class GroundednessReport:
    """Báo cáo đánh giá tính trung thực / căn cứ (Groundedness / Faithfulness)."""
    score: float             # Tỷ lệ mệnh đề có căn cứ / tổng số mệnh đề [0.0 -> 1.0]
    claims: List[ClaimVerification] = field(default_factory=list)
    is_grounded: bool = True # True nếu score >= 0.7 (không ảo giác)
    judge_reasoning: str = ""


@dataclass
class ContextRelevanceReport:
    """Báo cáo đánh giá mức độ liên quan của ngữ cảnh truy xuất (Context Relevance)."""
    score: float             # Điểm liên quan trung bình của các chunk [0.0 -> 1.0]
    relevant_chunks_count: int = 0
    total_chunks_count: int = 0
    chunk_scores: List[Dict[str, Any]] = field(default_factory=list)
    judge_reasoning: str = ""


@dataclass
class AnswerRelevanceReport:
    """Báo cáo đánh giá mức độ phù hợp và thỏa đáng của câu trả lời (Answer Relevance)."""
    score: float             # Điểm giải quyết đúng trọng tâm câu hỏi [0.0 -> 1.0]
    is_relevant: bool = True # True nếu score >= 0.7
    judge_reasoning: str = ""

