"""
Kịch bản thực nghiệm đối chứng:
SO SÁNH STANDARD RAG vs. NO-RAG vs. ASTUTE RAG
(Dựa trên bài báo 'ASTUTE RAG: Overcoming Imperfect Retrieval Augmentation and Knowledge Conflicts', ACL 2025)

Mục đích của kịch bản này:
1. Cho bạn thấy rõ hạn chế lớn nhất của Standard RAG: khi ngữ cảnh bên ngoài bị nhiễu hoặc sai lệch,
   Standard RAG sẽ làm suy giảm hiệu năng của LLM (thậm chí trả lời tệ hơn cả khi không dùng RAG - No RAG).
2. Minh họa cách ASTUTE RAG giải quyết triệt để vấn đề này qua cơ chế hợp nhất có nhận thức nguồn gốc (Source-aware Consolidation).
"""

from config import SimpleRAGConfig
from schema import EvalSample
from main_pipeline import SimpleRAGPipeline
from evaluation.generation_metrics import SimpleGenerationEvaluator
from analysis_astute_comparison.conflict_simulator import NoiseAndConflictSimulator
from analysis_astute_comparison.astute_rag_workflow import AstuteRAGConsolidator

def run_experiment():
    print("=====================================================================")
    print(" BÀI THỰC NGHIỆM: ĐÁNH GIÁ HẠN CHẾ CỦA STANDARD RAG SO VỚI ASTUTE RAG ")
    print("=====================================================================")

    config = SimpleRAGConfig()
    pipeline = SimpleRAGPipeline(config)
    evaluator = SimpleGenerationEvaluator()
    simulator = NoiseAndConflictSimulator()
    astute_consolidator = AstuteRAGConsolidator(config)

    # 1. Định nghĩa một câu hỏi thử nghiệm chứa xung đột tri thức (Lấy từ Ví dụ Hình 1 của bài báo):
    # Câu hỏi: Chính sách Một con của Trung Quốc chính thức kết thúc vào ngày nào?
    # - Tri thức đúng (Ground Truth): Ngày 01/01/2016.
    # - LLM nội tại nhớ: Ngày 01/01/2016 (Đúng).
    # - Tài liệu truy xuất ngoài (Nhiễu/Mâu thuẫn): "Trung Quốc thông báo kết thúc chính sách vào cuối năm 2015..."
    sample_query = "Chính sách Một con của Trung Quốc kết thúc vào thời gian nào?"
    ground_truth = "01/01/2016"
    misleading_passage = "Trung Quốc đã tuyên bố chấm dứt chính sách một con vào cuối năm 2015."

    print(f"\n[Câu hỏi kiểm thử]: {sample_query}")
    print(f"[Đáp án chuẩn (Ground Truth)]: {ground_truth}")

    # =====================================================================
    # CHẾ ĐỘ 1: NO-RAG (Chỉ dùng trí nhớ nội tại của LLM)
    # =====================================================================
    print("\n--- 1. Chạy chế độ NO-RAG (LLM Parametric Memory) ---")
    no_rag_output = pipeline.generator.generate_without_rag(sample_query)
    print(f"-> Đáp án NO-RAG: {no_rag_output.answer}")

    # =====================================================================
    # CHẾ ĐỘ 2: STANDARD RAG (Khi gặp tài liệu mâu thuẫn/nhiễu)
    # =====================================================================
    print("\n--- 2. Chạy chế độ STANDARD RAG (Ngữ cảnh bị nhiễu / xung đột) ---")
    # Mô phỏng tình huống retrieval mang về đoạn văn bản gây hiểu lầm:
    simulated_results = simulator.simulate_misleading_conflict(sample_query, ground_truth, misleading_passage)
    standard_rag_output = pipeline.generator.generate(sample_query, simulated_results)
    print(f"-> Đáp án STANDARD RAG: {standard_rag_output.answer}")
    
    # * NHẬN XÉT HỌC TẬP (The Pitfall of RAG):
    # LLM thường sẽ bị đoạn văn bản sai lừa (Confirmation Bias) và trả lời "cuối năm 2015".
    # Như vậy: Standard RAG bị điểm 0, trong khi No-RAG được điểm 1!
    # Thêm RAG vào lại làm hệ thống dở hơn!

    # =====================================================================
    # CHẾ ĐỘ 3: ASTUTE RAG (Hợp nhất có nhận thức nguồn gốc)
    # =====================================================================
    print("\n--- 3. Chạy chế độ ASTUTE RAG (Giải quyết xung đột tri thức) ---")
    astute_output = astute_consolidator.run_astute_rag(sample_query, simulated_results)
    print(f"-> Đáp án ASTUTE RAG: {astute_output.answer}")
    
    # * CÁCH ASTUTE RAG THẮNG ĐƯỢC:
    # - Bước 1: Cho LLM tự sinh ra tri thức nội tại của nó: "Kết thúc ngày 01/01/2016" [Nguồn: Memory].
    # - Bước 2: Nhận diện mâu thuẫn:
    #   Nhóm A (Memory): "Hiệu lực từ 01/01/2016".
    #   Nhóm B (External): "Thông báo vào cuối năm 2015".
    # - Bước 3: Đánh giá độ tin cậy và nhận ra 2015 chỉ là thời điểm công bố, còn 01/01/2016 mới là ngày chính thức kết thúc!
    # -> ASTUTE RAG đưa ra đáp án chính xác 01/01/2016.

    print("\n[*] Bạn hãy mở các file trong project simple_rag để hoàn thiện phần code bên trong các class!")

if __name__ == "__main__":
    import sys
    if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
        sys.stdout.reconfigure(encoding="utf-8")
    run_experiment()
