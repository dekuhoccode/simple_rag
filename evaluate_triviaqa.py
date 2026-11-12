"""
=====================================================================
KỊCH BẢN ĐÁNH GIÁ SIMPLE RAG TRÊN BỘ DỮ LIỆU CHUẨN TRIVIAQA
=====================================================================
Kịch bản này thực hiện quy trình đánh giá chuẩn học thuật cho Simple RAG:
1. Tải hoặc tái sử dụng bộ dữ liệu TriviaQA (mandarjoshi/trivia_qa - rc.wikipedia).
2. Chia nhỏ tài liệu (Chunking) theo chuẩn Simple RAG (fixed-size sliding window).
3. Đánh chỉ mục Vector vào SQLite Store & nhúng bằng Gemini Embedding (có cache chống tốn quota).
4. Truy xuất Top-K ngữ cảnh bằng Cosine Similarity (SimpleRetriever).
5. Sinh câu trả lời ngắn gọn (Short Extractive Answer) bằng Gemini LLM.
6. Chấm điểm toàn diện:
   - Tầng Retrieval: Hit Rate@K, MRR@K, Recall@K (SimpleRetrievalEvaluator).
   - Tầng Generation: Exact Match (EM) và F1-Score chuẩn TriviaQA.
   - (Tùy chọn) So sánh đối chứng với mô hình No-RAG (không dùng RAG).
   - (Tùy chọn) Chấm điểm RAG Triad (Groundedness, Relevance) qua LLM Judge.
"""

import os
import sys
import re
import json
import time
import string
import random
import pickle
import argparse
from collections import Counter
from typing import List, Dict, Any, Tuple, Optional

# Cấu hình UTF-8 cho Terminal Windows tránh lỗi UnicodeEncodeError cp1252
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if sys.stderr.encoding and sys.stderr.encoding.lower() != "utf-8":
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass
import numpy as np
from tqdm import tqdm

from config import SimpleRAGConfig
from schema import Document, Chunk, SearchResult, GenerationOutput, EvalSample, SourceType
from ingestion.document_loader import SimpleDocumentLoader
from chunking.text_chunker import SimpleTextChunker
from indexing.embedder import SimpleEmbedder
from indexing.vector_store import SimpleVectorStore
from retrieval.retriever import SimpleRetriever
from generation.prompt_builder import SimplePromptBuilder
from generation.llm_generator import SimpleLLMGenerator
from evaluation.retrieval_metrics import SimpleRetrievalEvaluator
from evaluation.generation_metrics import SimpleGenerationEvaluator
from evaluation.context_relevance_judge import ContextRelevanceJudge
from evaluation.answer_relevance_judge import AnswerRelevanceJudge
from evaluation.groundedness_judge import GroundednessJudge


# =====================================================================
# 1. PROMPT BUILDER CHUYÊN DỤNG CHO BENCHMARK TRIVIAQA
# =====================================================================
class TriviaQAPromptBuilder(SimplePromptBuilder):
    """
    Kế thừa SimplePromptBuilder để tối ưu hóa cho bài toán Question-Answering ngắn.
    Yêu cầu LLM trả lời ngắn gọn nhất (chỉ trả về tên thực thể, số, ngày tháng)
    để phục vụ việc đo đạc chỉ số Exact Match (EM) và F1-Score chuẩn xác.
    """
    def build_system_prompt(self) -> str:
        return (
            "You are a precise technical question-answering assistant.\n"
            "RULES:\n"
            "1. Answer the question using ONLY the provided context.\n"
            "2. Output ONLY the exact entity, name, number, date, or shortest possible phrase that directly answers the question.\n"
            "3. Do NOT write full sentences, explanations, conversational filler, or Markdown formatting.\n"
            "4. If the context does not contain enough information to answer, reply EXACTLY: \"I do not know\""
        )

    def build_user_prompt(self, query: str, search_results: List[SearchResult]) -> str:
        context_parts = [f"[Passage {i}] {sr.chunk.content}" for i, sr in enumerate(search_results, start=1)]
        context_text = "\n\n".join(context_parts)
        return (
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n"
            f"Concise Answer (exact entity/phrase only):"
        )


class TriviaQANoRAGPromptBuilder:
    """Prompt dành riêng cho kịch bản No-RAG (LLM trả lời chỉ dựa vào trí nhớ nội tại)."""
    @staticmethod
    def get_system_prompt() -> str:
        return (
            "You are a knowledgeable assistant.\n"
            "Answer the question directly and concisely.\n"
            "Output ONLY the exact entity, name, number, date, or shortest possible phrase.\n"
            "Do NOT write full sentences or explanations."
        )


# =====================================================================
# 2. CÁC HÀM CHẤM ĐIỂM CHUẨN TRIVIAQA (OFFICIAL EVALUATION METRICS)
# =====================================================================
def normalize_answer(s: str) -> str:
    """Chuẩn hóa chuỗi nguyên bản từ mã nguồn chính thức mandarjoshi90/triviaqa."""
    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def handle_punc(text: str) -> str:
        exclude = set(string.punctuation + "".join(["‘", "’", "´", "`"]))
        return "".join(ch if ch not in exclude else " " for ch in text)

    def lower(text: str) -> str:
        return text.lower()

    def replace_underscore(text: str) -> str:
        return text.replace("_", " ")

    return white_space_fix(remove_articles(handle_punc(lower(replace_underscore(s)))))


def f1_score_triviaqa(prediction: str, ground_truth: str) -> float:
    """Tính điểm F1-Score trên tập từ chuẩn hóa."""
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    return (2 * precision * recall) / (precision + recall)


def exact_match_score_triviaqa(prediction: str, ground_truth: str) -> float:
    """Tính điểm Exact Match (EM): 1.0 nếu chuỗi chuẩn hóa trùng khớp, ngược lại 0.0."""
    return 1.0 if normalize_answer(prediction) == normalize_answer(ground_truth) else 0.0


def metric_max_over_ground_truths(metric_fn, prediction: str, ground_truths: List[str]) -> float:
    """Lấy điểm cao nhất khi so sánh dự đoán với danh sách đáp án chuẩn và biệt danh (Aliases)."""
    scores = [metric_fn(prediction, gt) for gt in ground_truths]
    return max(scores) if scores else 0.0


# =====================================================================
# 3. TẢI HOẶC TÁI SỬ DỤNG BỘ DỮ LIỆU TRIVIAQA
# =====================================================================
def prepare_triviaqa_subset(data_dir: str, num_samples: int = 50, seed: int = 42) -> Tuple[str, str]:
    """
    Tải hoặc nạp tập dữ liệu TriviaQA.
    Tự động kiểm tra cả thư mục Custom-RAG lân cận để tái sử dụng, tránh tải lại tốn thời gian.
    """
    candidate_dirs = [
        data_dir,
        os.path.join("..", "Custom-RAG", data_dir),
        os.path.join("..", "Custom-RAG", "triviaqa_eval_data"),
        os.path.join("..", "triviaqa_eval_data"),
    ]

    for c_dir in candidate_dirs:
        dataset_candidate = os.path.join(c_dir, f"triviaqa_{num_samples}_dataset.json")
        corpus_candidate = os.path.join(c_dir, "corpus_md")
        if os.path.exists(dataset_candidate) and os.path.exists(corpus_candidate) and len(os.listdir(corpus_candidate)) > 0:
            print(f"[*] Tái sử dụng bộ dữ liệu có sẵn tại '{c_dir}' (bỏ qua bước tải lại).")
            return dataset_candidate, corpus_candidate

    os.makedirs(data_dir, exist_ok=True)
    corpus_dir = os.path.join(data_dir, "corpus_md")
    os.makedirs(corpus_dir, exist_ok=True)
    dataset_json_path = os.path.join(data_dir, f"triviaqa_{num_samples}_dataset.json")

    if os.path.exists(dataset_json_path) and len(os.listdir(corpus_dir)) > 0:
        print(f"[*] Đã tìm thấy bộ dữ liệu tại '{dataset_json_path}'.")
        return dataset_json_path, corpus_dir

    print(f"[*] Đang kết nối và stream tập 'mandarjoshi/trivia_qa' (rc.wikipedia) từ Hugging Face...")
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Vui lòng cài đặt thư viện datasets để tải dữ liệu: pip install datasets")

    ds = load_dataset("mandarjoshi/trivia_qa", "rc.wikipedia", split="validation", streaming=True)

    valid_pool = []
    target_pool_size = max(num_samples * 3, 300)
    for item in ds:
        if len(item.get("entity_pages", {}).get("wiki_context", [])) > 0:
            valid_pool.append(item)
        if len(valid_pool) >= target_pool_size:
            break

    random.seed(seed)
    sampled_items = random.sample(valid_pool, min(num_samples, len(valid_pool)))

    triviaqa_data_entries = []
    doc_registry: Dict[str, str] = {}

    for item in sampled_items:
        qid = item["question_id"]
        question = item["question"]
        ans = item["answer"]
        entity_pages = item["entity_pages"]

        sample_doc_ids = []
        combined_sections = []
        for title, wiki_ctx in zip(entity_pages["title"], entity_pages["wiki_context"]):
            combined_sections.append(f"# {title}\n\n{wiki_ctx}")

        if combined_sections:
            doc_id = f"doc{len(doc_registry) + 1:04d}"
            doc_registry[qid] = doc_id
            md_path = os.path.join(corpus_dir, f"{doc_id}.md")
            with open(md_path, "w", encoding="utf-8") as f:
                f.write("\n\n".join(combined_sections))
            sample_doc_ids.append(doc_id)

        triviaqa_data_entries.append({
            "QuestionId": qid,
            "Question": question,
            "TargetDocIds": sample_doc_ids,
            "Answer": {
                "Value": ans["value"],
                "NormalizedValue": ans["normalized_value"],
                "Aliases": ans["aliases"],
                "NormalizedAliases": ans["normalized_aliases"]
            }
        })

    official_format = {
        "Data": triviaqa_data_entries,
        "Domain": "Wikipedia",
        "VerifiedEval": False,
        "Version": 1.0
    }

    with open(dataset_json_path, "w", encoding="utf-8") as f:
        json.dump(official_format, f, ensure_ascii=False, indent=2)

    print(f"[*] Đã chuẩn bị thành công {len(triviaqa_data_entries)} câu hỏi và {len(doc_registry)} tài liệu tại '{corpus_dir}'.")
    return dataset_json_path, corpus_dir


# =====================================================================
# 4. CHUNKING & EMBEDDING CACHE CHO SIMPLE RAG
# =====================================================================
def build_or_load_vector_store(
    corpus_dir: str,
    cache_dir: str,
    config: SimpleRAGConfig,
    embedder: SimpleEmbedder,
    target_doc_ids: Optional[List[str]] = None
) -> Tuple[SimpleVectorStore, List[Chunk]]:
    """
    Thực hiện Chunking và Indexing tài liệu vào SimpleVectorStore.
    Có cache trung gian (chunks_cache.pkl và chunk_embeddings.npz) giúp tái sử dụng ngay lập tức.
    """
    os.makedirs(cache_dir, exist_ok=True)
    chunks_cache_path = os.path.join(cache_dir, "chunks_cache.pkl")

    model_safe_tag = "".join(c if c.isalnum() else "_" for c in config.embedding_model_name)
    vectors_cache_path = os.path.join(cache_dir, f"chunk_embeddings_{model_safe_tag}.npz")
    legacy_vectors_cache = os.path.join(cache_dir, "chunk_embeddings.npz")
    if not os.path.exists(vectors_cache_path) and os.path.exists(legacy_vectors_cache) and getattr(config, "embedding_provider", "gemini") == "gemini":
        vectors_cache_path = legacy_vectors_cache

    db_path = os.path.join(cache_dir, f"simple_rag_eval_{model_safe_tag}.db")
    legacy_db_path = os.path.join(cache_dir, "simple_rag_eval.db")
    if not os.path.exists(db_path) and os.path.exists(legacy_db_path) and getattr(config, "embedding_provider", "gemini") == "gemini":
        db_path = legacy_db_path

    vector_store = SimpleVectorStore(storage_path=db_path)

    # 1. Cắt đoạn (Chunking) bằng SimpleTextChunker
    if os.path.exists(chunks_cache_path):
        with open(chunks_cache_path, "rb") as f:
            all_chunks: List[Chunk] = pickle.load(f)
    else:
        chunker = SimpleTextChunker(chunk_size=config.chunk_size, chunk_overlap=config.chunk_overlap)
        loader = SimpleDocumentLoader()

        md_files = sorted([f for f in os.listdir(corpus_dir) if f.endswith(".md")])
        if target_doc_ids:
            target_set = set(target_doc_ids)
            md_files = [f for f in md_files if os.path.splitext(f)[0] in target_set]

        all_chunks = []
        for fname in tqdm(md_files, desc="Chunking Simple RAG"):
            fpath = os.path.join(corpus_dir, fname)
            doc = loader.load_file(fpath)
            chunks = chunker.chunk_document(doc)
            all_chunks.extend(chunks)

        with open(chunks_cache_path, "wb") as f:
            pickle.dump(all_chunks, f)

    # Nếu Vector Store SQLite đã có đủ chunks thì dùng luôn
    if vector_store.size >= len(all_chunks) and len(all_chunks) > 0:
        print(f"[*] Vector Store đã sẵn sàng với {vector_store.size} chunks từ SQLite cache.")
        return vector_store, all_chunks

    # 2. Quản lý Vector Embeddings Cache (tránh gọi API lặp lại)
    embedded_map: Dict[str, np.ndarray] = {}
    if os.path.exists(vectors_cache_path):
        data = np.load(vectors_cache_path, allow_pickle=True)
        for cid, vec in zip(data["chunk_ids"], data["vectors"]):
            embedded_map[str(cid)] = vec

    missing_chunks = [c for c in all_chunks if c.chunk_id not in embedded_map]
    batch_size = getattr(embedder, "max_batch_size", 32)

    if missing_chunks:
        provider_label = getattr(embedder, "provider", "gemini").upper()
        print(f"[*] Cần nhúng vector cho {len(missing_chunks)} chunks mới bằng {provider_label} Embedder ({embedder.model_name})...")
        for i in tqdm(range(0, len(missing_chunks), batch_size), desc=f"Batch Embedding ({provider_label})"):
            batch = missing_chunks[i:i + batch_size]
            t_start = time.time()
            vecs = embedder.embed_chunks(batch)
            for c, v in zip(batch, vecs):
                embedded_map[c.chunk_id] = v

            # Lưu checkpoint embeddings định kỳ
            np.savez_compressed(
                vectors_cache_path,
                chunk_ids=np.array(list(embedded_map.keys())),
                vectors=np.vstack(list(embedded_map.values())).astype(np.float32)
            )

            # Nghỉ nhịp tối thiểu 4s nếu dùng Gemini để tránh chạm rate limit (15 RPM)
            if getattr(embedder, "provider", "gemini") == "gemini":
                elapsed = time.time() - t_start
                min_delay = 4.0
                if elapsed < min_delay and i + batch_size < len(missing_chunks):
                    time.sleep(min_delay - elapsed)

    # 3. Nạp toàn bộ vào Vector Store
    chunks_to_add = []
    vectors_to_add = []
    for c in all_chunks:
        if c.chunk_id in embedded_map:
            c.embedding = embedded_map[c.chunk_id]
            chunks_to_add.append(c)
            vectors_to_add.append(embedded_map[c.chunk_id])

    vector_store.add_chunks(chunks_to_add, np.vstack(vectors_to_add))
    print(f"[*] Đã nạp thành công {vector_store.size} chunks vào SimpleVectorStore.")
    return vector_store, all_chunks


# =====================================================================
# 5. QUY TRÌNH ĐÁNH GIÁ CHÍNH (MAIN BENCHMARK PIPELINE)
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description="Đánh giá hiệu năng của Simple RAG trên benchmark TriviaQA")
    parser.add_argument("--num-samples", type=int, default=50, help="Số lượng câu hỏi đánh giá (mặc định: 50)")
    parser.add_argument("--data-dir", type=str, default="triviaqa_eval_data", help="Thư mục chứa dữ liệu và cache")
    parser.add_argument("--top-k", type=int, default=3, help="Số đoạn truy xuất Top-K của Simple RAG (mặc định: 3)")
    parser.add_argument("--compare-no-rag", action="store_true", help="Bật chế độ đối chứng song song với No-RAG (trí nhớ nội tại LLM)")
    parser.add_argument("--sleep-between-calls", type=float, default=1.0, help="Thời gian nghỉ (giây) giữa các câu hỏi để tránh Rate Limit")
    parser.add_argument("--embedding-provider", type=str, default=None, help="Nhà cung cấp embedding ('gemini', 'local', 'ollama')")
    parser.add_argument("--embedding-model", type=str, default=None, help="Tên mô hình embedding")
    parser.add_argument("--embedding-device", type=str, default=None, help="Thiết bị chạy local ('cpu' hoặc 'cuda')")
    parser.add_argument("--eval-rag-triad", "--run-judge", dest="eval_rag_triad", action="store_true", help="Kích hoạt chấm điểm RAG Triad (Context Relevance, Answer Relevance, Groundedness) qua LLM Judge")
    args = parser.parse_args()

    print("=" * 65)
    print(" KHỞI CHẠY ĐÁNH GIÁ HỆ THỐNG SIMPLE RAG TRÊN BỘ DỮ LIỆU TRIVIAQA ")
    print("=" * 65)

    config = SimpleRAGConfig()
    config.top_k = args.top_k
    if args.embedding_provider:
        config.embedding_provider = args.embedding_provider
    if args.embedding_model:
        config.embedding_model_name = args.embedding_model
    if args.embedding_device:
        config.embedding_device = args.embedding_device

    if config.embedding_provider == "gemini":
        if not config.gemini_api_key or "YOUR_" in config.gemini_api_key:
            print("[!] CẢNH BÁO: Chưa tìm thấy GEMINI_API_KEY trong file .env hoặc biến môi trường!")
            print("[!] GEMINI_API_KEY là bắt buộc khi dùng Embedder Gemini (models/gemini-embedding-2).")
            print("[!] Gợi ý: Đặt EMBEDDING_PROVIDER=local trong .env để chạy offline miễn phí không cần Gemini Key!")
            return

    if not config.llm_api_key or "YOUR_" in config.llm_api_key:
        print(f"[!] CẢNH BÁO: Chưa tìm thấy APIKEY / LLM_API_KEY trong file .env!")
        print(f"[!] Vui lòng cấu hình API Key cho mô hình sinh ({config.llm_model} tại {config.llm_base_url}).")
        return

    # 1. Tải và chuẩn bị bộ dữ liệu TriviaQA
    dataset_path, corpus_dir = prepare_triviaqa_subset(args.data_dir, num_samples=args.num_samples)
    with open(dataset_path, "r", encoding="utf-8") as f:
        triviaqa_items = json.load(f)["Data"]

    target_docs = []
    for item in triviaqa_items:
        target_docs.extend(item.get("TargetDocIds", []))
    target_docs = list(set(target_docs))

    # 2. Khởi tạo Embedder & Quản lý Cache Vector
    cache_dir = os.path.join(args.data_dir, "cache_simple_rag")
    os.makedirs(cache_dir, exist_ok=True)
    
    model_safe_tag = "".join(c if c.isalnum() else "_" for c in config.embedding_model_name)
    query_cache_path = os.path.join(cache_dir, f"query_embeddings_{model_safe_tag}.pkl")
    legacy_query_cache = os.path.join(cache_dir, "query_embeddings.pkl")
    if not os.path.exists(query_cache_path) and os.path.exists(legacy_query_cache) and config.embedding_provider == "gemini":
        query_cache_path = legacy_query_cache

    embedder = SimpleEmbedder(
        provider=config.embedding_provider,
        model_name=config.embedding_model_name,
        api_key=config.gemini_api_key,
        output_dimensionality=config.embedding_dim,
        device=getattr(config, "embedding_device", "cpu"),
        ollama_base_url=getattr(config, "ollama_base_url", "http://localhost:11434")
    )

    query_cache: Dict[str, np.ndarray] = {}
    if os.path.exists(query_cache_path):
        with open(query_cache_path, "rb") as f:
            query_cache = pickle.load(f)

    # 3. Xây dựng SimpleVectorStore
    vector_store, all_chunks = build_or_load_vector_store(
        corpus_dir=corpus_dir,
        cache_dir=cache_dir,
        config=config,
        embedder=embedder,
        target_doc_ids=target_docs
    )

    # 4. Khởi tạo Retriever, PromptBuilder và LLMGenerator
    retriever = SimpleRetriever(
        embedder=embedder,
        vector_store=vector_store,
        top_k=config.top_k
    )

    prompt_builder = TriviaQAPromptBuilder()
    llm_generator = SimpleLLMGenerator(
        model_name=config.llm_model,
        api_key=config.llm_api_key,
        base_url=config.llm_base_url,
        fallback_model_name=config.fallback_llm_model,
        temperature=config.temperature,
        max_rpm=config.max_rpm,
        prompt_builder=prompt_builder
    )

    retrieval_evaluator = SimpleRetrievalEvaluator()

    # Quản lý Checkpoints dự đoán
    predictions_path = os.path.join(args.data_dir, "simple_rag_predictions.json")
    no_rag_predictions_path = os.path.join(args.data_dir, "no_rag_predictions.json")

    predictions: Dict[str, str] = {}
    if os.path.exists(predictions_path):
        with open(predictions_path, "r", encoding="utf-8") as f:
            predictions = json.load(f)

    no_rag_predictions: Dict[str, str] = {}
    if os.path.exists(no_rag_predictions_path):
        with open(no_rag_predictions_path, "r", encoding="utf-8") as f:
            no_rag_predictions = json.load(f)

    eval_samples: List[EvalSample] = []
    retrieved_results_per_sample: List[List[SearchResult]] = []

    print(f"\n[*] Bắt đầu thực thi kiểm thử trên {len(triviaqa_items)} câu hỏi...")
    for item in tqdm(triviaqa_items, desc="Evaluating Simple RAG"):
        qid = item["QuestionId"]
        query = item["Question"]
        ans_info = item["Answer"]

        sample = EvalSample(
            sample_id=qid,
            query=query,
            ground_truth_answer=ans_info["Value"],
            target_keywords=[ans_info["Value"]] + (ans_info.get("Aliases") or [])[:3]
        )

        try:
            # Giai đoạn Retrieval: Tìm Top-K
            # Sử dụng query cache nếu có
            if query in query_cache:
                q_vec = query_cache[query]
                search_results = vector_store.similarity_search(q_vec, top_k=config.top_k)
            else:
                q_vec = embedder.embed_query(query)
                query_cache[query] = q_vec
                with open(query_cache_path, "wb") as f:
                    pickle.dump(query_cache, f)
                search_results = vector_store.similarity_search(q_vec, top_k=config.top_k)

            # Giai đoạn Generation: Simple RAG
            # Tự động thử lại nếu kết quả cache là lỗi API trước đó (ví dụ [Lỗi gọi LLM API...])
            if qid in predictions and not str(predictions[qid]).startswith("[Lỗi"):
                pred_answer = predictions[qid]
            else:
                gen_output = llm_generator.generate(query, search_results)
                pred_answer = gen_output.answer.strip()
                predictions[qid] = pred_answer
                with open(predictions_path, "w", encoding="utf-8") as f:
                    json.dump(predictions, f, ensure_ascii=False, indent=2)

                if args.sleep_between_calls > 0:
                    time.sleep(args.sleep_between_calls)

            # (Tùy chọn) Chạy thêm kịch bản No-RAG đối chứng
            if args.compare_no_rag:
                if qid in no_rag_predictions and not str(no_rag_predictions[qid]).startswith("[Lỗi"):
                    no_rag_ans = no_rag_predictions[qid]
                else:
                    no_rag_out = llm_generator.generate_without_rag(
                        query,
                        system_prompt=TriviaQANoRAGPromptBuilder.get_system_prompt()
                    )
                    no_rag_ans = no_rag_out.answer.strip()
                    no_rag_predictions[qid] = no_rag_ans
                    with open(no_rag_predictions_path, "w", encoding="utf-8") as f:
                        json.dump(no_rag_predictions, f, ensure_ascii=False, indent=2)

            eval_samples.append(sample)
            retrieved_results_per_sample.append(search_results)

        except Exception as e:
            print(f"\n[!] Dừng sớm tại câu hỏi '{qid}' do lỗi / chạm rate limit: {e}")
            break

    if not eval_samples:
        print("[!] Chưa có mẫu nào hoàn tất để tổng hợp báo cáo.")
        return

    # =====================================================================
    # 6. TỔNG HỢP VÀ IN BÁO CÁO KẾT QUẢ
    # =====================================================================
    print("\n" + "=" * 65)
    print(f" BÁO CÁO HIỆU NĂNG SIMPLE RAG ({len(eval_samples)}/{len(triviaqa_items)} CÂU)")
    print("=" * 65)

    # 1. Đánh giá chất lượng Retrieval (0 API call)
    ret_summary = retrieval_evaluator.evaluate_dataset(
        samples=eval_samples,
        results_list=retrieved_results_per_sample,
        k=config.top_k
    )
    print(f"\n[1] CHỈ SỐ TRUY XUẤT (RETRIEVAL METRICS @ Top-{config.top_k}):")
    print(f"  - Hit Rate@{config.top_k} : {ret_summary['mean_hit_rate'] * 100:.2f}%")
    print(f"  - MRR@{config.top_k}      : {ret_summary['mean_mrr']:.4f}")
    print(f"  - Recall@{config.top_k}   : {ret_summary['mean_recall'] * 100:.2f}%")

    # 2. Đánh giá chất lượng Generation của Simple RAG (Official EM & F1)
    em_total, f1_total = 0.0, 0.0
    item_by_qid = {it["QuestionId"]: it for it in triviaqa_items}

    for sample in eval_samples:
        qid = sample.sample_id
        pred_str = predictions.get(qid, "")
        gt_aliases = item_by_qid[qid]["Answer"]["Aliases"] + [item_by_qid[qid]["Answer"]["Value"]]

        em_total += metric_max_over_ground_truths(exact_match_score_triviaqa, pred_str, gt_aliases)
        f1_total += metric_max_over_ground_truths(f1_score_triviaqa, pred_str, gt_aliases)

    n_eval = len(eval_samples)
    print(f"\n[2] CHỈ SỐ SINH CÂU TRẢ LỜI CỦA SIMPLE RAG (GENERATION METRICS):")
    print(f"  - Exact Match (EM): {(em_total / n_eval) * 100:.2f}%")
    print(f"  - F1 Score        : {(f1_total / n_eval) * 100:.2f}%")

    # 3. So sánh với No-RAG (nếu người dùng bật cờ --compare-no-rag)
    if args.compare_no_rag and no_rag_predictions:
        no_rag_em, no_rag_f1 = 0.0, 0.0
        for sample in eval_samples:
            qid = sample.sample_id
            pred_str = no_rag_predictions.get(qid, "")
            gt_aliases = item_by_qid[qid]["Answer"]["Aliases"] + [item_by_qid[qid]["Answer"]["Value"]]
            no_rag_em += metric_max_over_ground_truths(exact_match_score_triviaqa, pred_str, gt_aliases)
            no_rag_f1 += metric_max_over_ground_truths(f1_score_triviaqa, pred_str, gt_aliases)

        print(f"\n[3] ĐỐI CHỨNG VỚI NO-RAG (PARAMETRIC MEMORY):")
        print(f"  - No-RAG Exact Match (EM) : {(no_rag_em / n_eval) * 100:.2f}%")
        print(f"  - No-RAG F1 Score         : {(no_rag_f1 / n_eval) * 100:.2f}%")
        print(f"  - Chênh lệch EM (RAG - No-RAG): {((em_total - no_rag_em) / n_eval) * 100:+.2f}%")

    # 4. Đánh giá chất lượng RAG Triad (nếu bật --eval-rag-triad hoặc --run-judge)
    rag_triad_summary = {}
    sample_triad_details = {}
    if args.eval_rag_triad:
        print(f"\n[4] ĐANG ĐÁNH GIÁ RAG TRIAD (LLM-AS-A-JUDGE)...")
        ctx_judge = ContextRelevanceJudge(config=config)
        ans_judge = AnswerRelevanceJudge(config=config)
        grd_judge = GroundednessJudge(config=config)

        total_ctx_rel = 0.0
        total_ans_rel = 0.0
        total_grd = 0.0
        triad_count = 0

        for sample, s_results in tqdm(zip(eval_samples, retrieved_results_per_sample), total=len(eval_samples), desc="Evaluating RAG Triad"):
            qid = sample.sample_id
            pred_str = predictions.get(qid, "")

            ctx_report = ctx_judge.evaluate(sample.query, s_results)
            c_score = ctx_report.score

            if pred_str.startswith("[Lỗi"):
                a_score = 0.0
                g_score = 0.0
            else:
                ans_report = ans_judge.evaluate(sample.query, pred_str, ground_truth=sample.ground_truth_answer)
                grd_report = grd_judge.evaluate(pred_str, s_results)
                a_score = ans_report.score
                g_score = grd_report.score

            total_ctx_rel += c_score
            total_ans_rel += a_score
            total_grd += g_score
            triad_count += 1

            sample_triad_details[qid] = {
                "context_relevance": round(c_score, 4),
                "answer_relevance": round(a_score, 4),
                "groundedness": round(g_score, 4)
            }

        mean_ctx_rel = total_ctx_rel / triad_count if triad_count > 0 else 0.0
        mean_ans_rel = total_ans_rel / triad_count if triad_count > 0 else 0.0
        mean_grd = total_grd / triad_count if triad_count > 0 else 0.0

        rag_triad_summary = {
            "context_relevance": round(mean_ctx_rel * 100, 2),
            "answer_relevance": round(mean_ans_rel * 100, 2),
            "groundedness": round(mean_grd * 100, 2)
        }

        print(f"  - Context Relevance      : {rag_triad_summary['context_relevance']}% (Độ liên quan của ngữ cảnh truy xuất)")
        print(f"  - Answer Relevance       : {rag_triad_summary['answer_relevance']}% (Độ thỏa đáng của câu trả lời)")
        print(f"  - Groundedness (Trung thực): {rag_triad_summary['groundedness']}% (Không ảo giác)")
    else:
        print(f"\n[*] GỢI Ý: Chạy kèm cờ '--eval-rag-triad' (hoặc '--run-judge') để đo đạc và in 3 chỉ số RAG Triad:")
        print(f"    - Context Relevance, Answer Relevance, Groundedness (LLM-as-a-Judge).")

    # =====================================================================
    # 7. XUẤT TOÀN BỘ BÁO CÁO VÀO CÁC FILE LƯU TRỮ VĨNH VIỄN
    # =====================================================================
    report_json_path = os.path.join(args.data_dir, "evaluation_report.json")
    report_md_path = os.path.join(args.data_dir, "evaluation_report.md")

    per_sample_details = []
    for sample, s_results in zip(eval_samples, retrieved_results_per_sample):
        qid = sample.sample_id
        pred_str = predictions.get(qid, "")
        gt_aliases = item_by_qid[qid]["Answer"]["Aliases"] + [item_by_qid[qid]["Answer"]["Value"]]

        em = metric_max_over_ground_truths(exact_match_score_triviaqa, pred_str, gt_aliases)
        f1 = metric_max_over_ground_truths(f1_score_triviaqa, pred_str, gt_aliases)

        sample_detail = {
            "question_id": qid,
            "question": sample.query,
            "ground_truth_answer": item_by_qid[qid]["Answer"]["Value"],
            "aliases": item_by_qid[qid]["Answer"]["Aliases"],
            "retrieved_chunks": [
                {
                    "rank": r.rank,
                    "chunk_id": r.chunk.chunk_id,
                    "score": round(r.score, 4),
                    "snippet": r.chunk.content[:150] + "..." if len(r.chunk.content) > 150 else r.chunk.content
                }
                for r in s_results
            ],
            "simple_rag": {
                "predicted_answer": pred_str,
                "exact_match": em,
                "f1_score": round(f1, 4)
            }
        }

        if qid in sample_triad_details:
            sample_detail["rag_triad"] = sample_triad_details[qid]

        if args.compare_no_rag and no_rag_predictions:
            no_rag_str = no_rag_predictions.get(qid, "")
            n_em = metric_max_over_ground_truths(exact_match_score_triviaqa, no_rag_str, gt_aliases)
            n_f1 = metric_max_over_ground_truths(f1_score_triviaqa, no_rag_str, gt_aliases)
            sample_detail["no_rag"] = {
                "predicted_answer": no_rag_str,
                "exact_match": n_em,
                "f1_score": round(n_f1, 4)
            }

        per_sample_details.append(sample_detail)

    final_report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "configuration": {
            "num_samples_evaluated": n_eval,
            "top_k": config.top_k,
            "embedding_model": config.embedding_model_name,
            "llm_model": config.llm_model,
            "llm_base_url": config.llm_base_url,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap
        },
        "retrieval_metrics": {
            "hit_rate": round(ret_summary["mean_hit_rate"] * 100, 2),
            "mrr": round(ret_summary["mean_mrr"], 4),
            "recall": round(ret_summary["mean_recall"] * 100, 2)
        },
        "generation_metrics": {
            "simple_rag": {
                "exact_match": round((em_total / n_eval) * 100, 2),
                "f1_score": round((f1_total / n_eval) * 100, 2)
            }
        },
        "detailed_results": per_sample_details
    }

    if rag_triad_summary:
        final_report["rag_triad_metrics"] = rag_triad_summary

    if args.compare_no_rag and no_rag_predictions:
        final_report["generation_metrics"]["no_rag"] = {
            "exact_match": round((no_rag_em / n_eval) * 100, 2),
            "f1_score": round((no_rag_f1 / n_eval) * 100, 2)
        }
        final_report["generation_metrics"]["delta_exact_match"] = round(((em_total - no_rag_em) / n_eval) * 100, 2)

    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(final_report, f, ensure_ascii=False, indent=2)

    # Xuất file Markdown trực quan
    md_lines = [
        "# Báo Cáo Đánh Giá Hiệu Năng Simple RAG (TriviaQA Benchmark)",
        f"- **Thời gian thực hiện:** {final_report['timestamp']}",
        f"- **Số lượng câu hỏi:** {n_eval} / {len(triviaqa_items)}",
        f"- **Mô hình Embedding:** `{config.embedding_model_name}` (Top-K={config.top_k})",
        f"- **Mô hình Sinh (LLM):** `{config.llm_model}` ({config.llm_base_url})",
        "",
        "## 1. Kết Quả Tổng Hợp (Summary Metrics)",
        "",
        "### A. Tầng Truy Xuất (Retrieval Metrics)",
        f"| Chỉ số | Điểm số | Ý nghĩa |",
        f"| :--- | :--- | :--- |",
        f"| **Hit Rate@{config.top_k}** | `{final_report['retrieval_metrics']['hit_rate']}%` | Tỷ lệ câu hỏi tìm thấy đáp án trong Top-{config.top_k} |",
        f"| **MRR@{config.top_k}** | `{final_report['retrieval_metrics']['mrr']}` | Thứ hạng đảo trung bình của kết quả đúng đầu tiên |",
        f"| **Recall@{config.top_k}** | `{final_report['retrieval_metrics']['recall']}%` | Tỷ lệ bao phủ từ khóa đáp án |",
        "",
        "### B. Tầng Sinh Câu Trả Lời (Generation Metrics)",
        f"| Mô hình | Exact Match (EM) | F1-Score |",
        f"| :--- | :--- | :--- |",
        f"| **Simple RAG** | **{final_report['generation_metrics']['simple_rag']['exact_match']}%** | **{final_report['generation_metrics']['simple_rag']['f1_score']}%** |"
    ]
    if args.compare_no_rag and no_rag_predictions:
        md_lines.append(f"| **No-RAG (Parametric)** | {final_report['generation_metrics']['no_rag']['exact_match']}% | {final_report['generation_metrics']['no_rag']['f1_score']}% |")
        md_lines.append(f"| **Chênh lệch (RAG - No-RAG)** | **{final_report['generation_metrics']['delta_exact_match']:+}%** | - |")

    if rag_triad_summary:
        md_lines.extend([
            "",
            "### C. Tầng Đánh Giá Chất Lượng Toàn Diện (RAG Triad Metrics)",
            "| Chỉ số | Điểm số | Ý nghĩa |",
            "| :--- | :--- | :--- |",
            f"| **Context Relevance** | `{rag_triad_summary['context_relevance']}%` | Tỷ lệ ngữ cảnh truy xuất thực sự hữu ích cho câu hỏi |",
            f"| **Answer Relevance** | `{rag_triad_summary['answer_relevance']}%` | Mức độ phù hợp và thỏa đáng của câu trả lời |",
            f"| **Groundedness (Faithfulness)** | `{rag_triad_summary['groundedness']}%` | Mức độ trung thực dựa trên tài liệu, không ảo giác |",
        ])

    md_lines.extend([
        "",
        "## 2. Chi Tiết Từng Câu Hỏi Kiểm Thử (Samples Breakdown)",
        ""
    ])
    for idx, s in enumerate(per_sample_details, start=1):
        md_lines.extend([
            f"### #{idx}. {s['question']}",
            f"- **Đáp án chuẩn (Ground Truth):** `{s['ground_truth_answer']}`",
            f"- **Simple RAG:** `{s['simple_rag']['predicted_answer']}` (EM={s['simple_rag']['exact_match']}, F1={s['simple_rag']['f1_score']})",
        ])
        if "rag_triad" in s:
            md_lines.append(f"- **RAG Triad:** Context Rel={s['rag_triad']['context_relevance']}, Answer Rel={s['rag_triad']['answer_relevance']}, Groundedness={s['rag_triad']['groundedness']}")
        if "no_rag" in s:
            md_lines.append(f"- **No-RAG:** `{s['no_rag']['predicted_answer']}` (EM={s['no_rag']['exact_match']}, F1={s['no_rag']['f1_score']})")
        md_lines.append("")

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    print("\n" + "=" * 65)
    print(f"[*] ĐÃ LƯU TOÀN BỘ KẾT QUẢ ĐÁNH GIÁ VÀO:")
    print(f"    1. Báo cáo Markdown trực quan : {report_md_path}")
    print(f"    2. Dữ liệu JSON chi tiết      : {report_json_path}")
    print(f"    3. File dự đoán Simple RAG    : {predictions_path}")
    if args.compare_no_rag:
        print(f"    4. File dự đoán No-RAG        : {no_rag_predictions_path}")
    print("=" * 65)


if __name__ == "__main__":
    main()
