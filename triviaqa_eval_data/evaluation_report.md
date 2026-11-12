# Báo Cáo Đánh Giá Hiệu Năng Simple RAG (TriviaQA Benchmark)
- **Thời gian thực hiện:** 2026-11-11 23:32:43
- **Số lượng câu hỏi:** 10 / 10
- **Mô hình Embedding:** `BAAI/bge-small-en-v1.5` (Top-K=3)
- **Mô hình Sinh (LLM):** `gemini-3.8-flash` (https://aiplatform.googleapis.com/v1/projects/my-first-rag-514304/locations/global/publishers/google/models)

## 1. Kết Quả Tổng Hợp (Summary Metrics)

### A. Tầng Truy Xuất (Retrieval Metrics)
| Chỉ số | Điểm số | Ý nghĩa |
| :--- | :--- | :--- |
| **Hit Rate@3** | `50.0%` | Tỷ lệ câu hỏi tìm thấy đáp án trong Top-3 |
| **MRR@3** | `0.5` | Thứ hạng đảo trung bình của kết quả đúng đầu tiên |
| **Recall@3** | `23.33%` | Tỷ lệ bao phủ từ khóa đáp án |

### B. Tầng Sinh Câu Trả Lời (Generation Metrics)
| Mô hình | Exact Match (EM) | F1-Score |
| :--- | :--- | :--- |
| **Simple RAG** | **40.0%** | **43.33%** |

### C. Tầng Đánh Giá Chất Lượng Toàn Diện (RAG Triad Metrics)
| Chỉ số | Điểm số | Ý nghĩa |
| :--- | :--- | :--- |
| **Context Relevance** | `44.9%` | Tỷ lệ ngữ cảnh truy xuất thực sự hữu ích cho câu hỏi |
| **Answer Relevance** | `41.0%` | Mức độ phù hợp và thỏa đáng của câu trả lời |
| **Groundedness (Faithfulness)** | `100.0%` | Mức độ trung thực dựa trên tài liệu, không ảo giác |

## 2. Chi Tiết Từng Câu Hỏi Kiểm Thử (Samples Breakdown)

### #1. Who first hosted Family Feud?
- **Đáp án chuẩn (Ground Truth):** `Richard Dawson`
- **Simple RAG:** `I do not know` (EM=0.0, F1=0.0)
- **RAG Triad:** Context Rel=0.55, Answer Rel=0.0, Groundedness=1.0

### #2. Who directed the classic 30s western Stagecoach?
- **Đáp án chuẩn (Ground Truth):** `John Ford`
- **Simple RAG:** `John Ford` (EM=1.0, F1=1.0)
- **RAG Triad:** Context Rel=0.45, Answer Rel=1.0, Groundedness=1.0

### #3. What are the international registration letters of a vehicle from Iceland?
- **Đáp án chuẩn (Ground Truth):** `IS`
- **Simple RAG:** `I do not know` (EM=0.0, F1=0.3333)
- **RAG Triad:** Context Rel=0.2, Answer Rel=0.0, Groundedness=1.0

### #4. In which North African country was Mohammed Al-Fayed, owner of Harrods department store in London, born?
- **Đáp án chuẩn (Ground Truth):** `Egypt`
- **Simple RAG:** `I do not know` (EM=0.0, F1=0.0)
- **RAG Triad:** Context Rel=0.05, Answer Rel=0.0, Groundedness=1.0

### #5. Which grand slam did Pete Sampras not win in the 20th century?
- **Đáp án chuẩn (Ground Truth):** `French`
- **Simple RAG:** `I do not know` (EM=0.0, F1=0.0)
- **RAG Triad:** Context Rel=0.35, Answer Rel=0.1, Groundedness=1.0

### #6. In which American state is the Isabella Stewart Gardner Museum?
- **Đáp án chuẩn (Ground Truth):** `Massachusetts`
- **Simple RAG:** `Massachusetts` (EM=1.0, F1=1.0)
- **RAG Triad:** Context Rel=0.67, Answer Rel=1.0, Groundedness=1.0

### #7. Who was the first person after Scott to reach the South Pole overland?
- **Đáp án chuẩn (Ground Truth):** `Edmond Hillary`
- **Simple RAG:** `I do not know` (EM=0.0, F1=0.0)
- **RAG Triad:** Context Rel=0.15, Answer Rel=0.0, Groundedness=1.0

### #8. Who bought Chelsea football club for £1 in 1982?
- **Đáp án chuẩn (Ground Truth):** `Ken Bates`
- **Simple RAG:** `Ken Bates` (EM=1.0, F1=1.0)
- **RAG Triad:** Context Rel=1.0, Answer Rel=1.0, Groundedness=1.0

### #9. River Phoenix died during the making of which movie?
- **Đáp án chuẩn (Ground Truth):** `Dark Blood`
- **Simple RAG:** `I do not know` (EM=0.0, F1=0.0)
- **RAG Triad:** Context Rel=0.15, Answer Rel=0.0, Groundedness=1.0

### #10. How many pieces are there in a standard set of dominoes?
- **Đáp án chuẩn (Ground Truth):** `28`
- **Simple RAG:** `28` (EM=1.0, F1=1.0)
- **RAG Triad:** Context Rel=0.92, Answer Rel=1.0, Groundedness=1.0
