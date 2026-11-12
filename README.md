# Dự án Simple RAG & Đối chứng ASTUTE RAG (Educational Framework)

Thư mục dự án này được thiết kế theo cấu trúc mô-đun hóa từng giai đoạn tiêu chuẩn của **RAG (Retrieval-Augmented Generation)**, đồng thời tích hợp khung thực nghiệm để chỉ ra **hạn chế của Standard RAG so với giải pháp ASTUTE RAG** (từ bài báo *ASTUTE RAG: Overcoming Imperfect Retrieval Augmentation and Knowledge Conflicts for Large Language Models*, ACL 2025).

---

## 1. Cấu trúc thư mục dự án

```text
simple_rag/
│
├── config.py                                 # Cấu hình siêu tham số (chunk_size, top_k, model, v.v.)
├── schema.py                                 # Định nghĩa các Data Class cốt lõi (Document, Chunk, SearchResult...)
├── main_pipeline.py                          # Luồng thực thi hoàn chỉnh End-to-End của Simple RAG
├── run_astute_comparison.py                  # Kịch bản thực nghiệm đối chứng: No-RAG vs Standard RAG vs ASTUTE RAG
│
├── ingestion/                                # GIAI ĐOẠN 1: Thu nạp dữ liệu
│   ├── __init__.py
│   └── document_loader.py                    # Class SimpleDocumentLoader: Đọc file txt/md, làm sạch text
│
├── chunking/                                 # GIAI ĐOẠN 2: Cắt nhỏ văn bản
│   ├── __init__.py
│   └── text_chunker.py                       # Class SimpleTextChunker: Cắt fixed-size sliding window
│
├── indexing/                                 # GIAI ĐOẠN 3: Đánh chỉ mục & Vector
│   ├── __init__.py
│   ├── embedder.py                           # Class SimpleEmbedder: Nhúng văn bản và câu hỏi thành Vector
│   └── vector_store.py                       # Class SimpleVectorStore: Lưu ma trận vector và tính Cosine Similarity
│
├── retrieval/                                # GIAI ĐOẠN 4: Truy xuất
│   ├── __init__.py
│   └── retriever.py                          # Class SimpleRetriever: Nhận câu hỏi, tìm Top-K chunk tương đồng
│
├── generation/                               # GIAI ĐOẠN 5: Sinh câu trả lời
│   ├── __init__.py
│   ├── prompt_builder.py                     # Class SimplePromptBuilder: Nối phẳng context và câu hỏi vào Prompt
│   └── llm_generator.py                      # Class SimpleLLMGenerator: Gọi LLM (có hàm generate và generate_without_rag)
│
├── evaluation/                               # GIAI ĐOẠN 6: Đánh giá chất lượng
│   ├── __init__.py
│   ├── retrieval_metrics.py                  # Class SimpleRetrievalEvaluator: Đo Hit Rate@K, Recall@K, MRR@K
│   └── generation_metrics.py                 # Class SimpleGenerationEvaluator: Đo EM, F1 và phân loại Knowledge Conflicts
│
└── analysis_astute_comparison/               # PHÂN TÍCH HẠN CHẾ SO VỚI ASTUTE RAG
    ├── __init__.py
    ├── conflict_simulator.py                 # Class NoiseAndConflictSimulator: Mô phỏng nhiễu 70% và xung đột tri thức
    └── astute_rag_workflow.py                # Class AstuteRAGConsolidator: Khung 3 bước giải quyết xung đột của ASTUTE RAG
```

---

## 2. Vì sao Standard RAG thất bại? (Đối chứng với bài báo ASTUTE RAG)

Khi bạn chạy và kiểm thử mô hình Simple/Standard RAG trên, bạn sẽ quan sát thấy các hạn chế cốt lõi mà bài báo của Google Cloud AI Research & USC đã công bố:

### A. Thực trạng "Truy xuất không hoàn hảo" (Imperfect Retrieval)
* **Standard RAG giả định:** Mọi tài liệu có điểm tương đồng Cosine cao nhất đều chứa đáp án đúng.
* **Thực tế nghiên cứu (Hình 2 trong bài báo):** Khoảng **70% các đoạn truy xuất** từ Web/Corpus thực tế là **nhiễu** (không chứa câu trả lời đúng, thông tin đã cũ, hoặc nói về một thực thể khác).
* **Hậu quả:** Standard RAG ném toàn bộ các đoạn này vào Prompt, làm phân tán sự chú ý và đầu độc ngữ cảnh của LLM.

### B. Mâu thuẫn tri thức & Hiện tượng suy giảm hiệu năng (The Pitfall of RAG)
* **Nghịch lý thực nghiệm (Bảng 1 trong bài báo):** Trên các tập dữ liệu như NQ và TriviaQA, mô hình **No-RAG (không dùng RAG)** lại đạt độ chính xác **cao hơn** so với **Standard RAG**!
* **Nguyên nhân:** Bản thân LLM (như Gemini, Claude, Llama) đã học rất nhiều kiến thức trong giai đoạn Pre-training. Khi gặp câu hỏi quen thuộc, LLM tự trả lời đúng. Nhưng khi Standard RAG cung cấp một đoạn văn bản trích xuất bên ngoài bị sai lệch hoặc mâu thuẫn:
  * LLM bị hiệu ứng **Confirmation Bias / Sycophancy**: Tin theo đoạn văn bản sai và bỏ rơi tri thức đúng sẵn có trong đầu.
  * Tỷ lệ xung đột tri thức lên tới gần **20%** trong thực tế.

### C. Cách ASTUTE RAG khắc phục:
1. **Adaptive Generation (Khơi gợi tri thức nội tại):** Trước khi tin hoàn toàn vào tài liệu ngoài, cho LLM tự phát biểu tri thức của chính nó với nguyên tắc "nếu không chắc thì nói không biết".
2. **Source-aware Consolidation (Hợp nhất theo nguồn):** Gom các đoạn đồng thuận thành cụm, tách riêng các đoạn mâu thuẫn, lọc bỏ đoạn rác và dán nhãn nguồn rõ ràng (`[Memory]` vs `[External]`).
3. **Answer Finalization with Confidence:** Đánh giá độ tin cậy dựa trên sự xác nhận chéo (Cross-source confirmation) để ra quyết định cuối cùng.

---

## 3. Hướng dẫn tiếp tục hoàn thiện code

Mỗi file Python trong các thư mục đều đã được thiết kế sẵn:
* Chữ ký hàm (`def method_name(...) -> Type:`)
* Docstring chi tiết giải thích rõ ràng **mục đích của class**, **đầu vào (Args)**, **đầu ra (Returns)**.
* Khối `# Gợi ý code` hoặc các bước `# TODO` chi tiết để bạn tự nghiên cứu và điền phần code thực thi.
