from typing import List
from schema import SearchResult

class SimplePromptBuilder:
    """
    Giai đoạn 5A: Generation - Đóng gói Prompt.
    
    Trong mô hình Simple RAG:
    - Nối (concatenate) phẳng tất cả các đoạn văn bản Top-K tìm được thành một khối ngữ cảnh (Context block).
    - Ghép chung với câu hỏi của người dùng và gửi cho LLM.
    
    * LƯU Ý HỌC TẬP & HẠN CHẾ SO VỚI BÀI BÁO ASTUTE RAG:
      1. Hiện tượng "Bị làm mù bởi ngữ cảnh" (Blinded by context / Confirmation Bias):
         Khi các đoạn văn bản được ném vào prompt một cách thô mộc, nếu trong đó có một đoạn sai lệch
         (misleading/incorrect), LLM thường có xu hướng bỏ qua tri thức đúng trong đầu mình để lặp lại
         thông tin sai trong context.
      2. Mâu thuẫn tri thức (Knowledge Conflicts):
         Nếu đoạn 1 nói "Sự kiện kết thúc năm 2015" và đoạn 2 nói "Sự kiện kết thúc năm 2016",
         Simple Prompt không hướng dẫn LLM cách phân xử, dẫn đến câu trả lời bị mâu thuẫn hoặc ảo giác.
      3. ASTUTE RAG giải quyết vấn đề này bằng cách:
         - Không đưa thô cả danh sách vào prompt cuối ngay.
         - Sử dụng cấu trúc nhóm (Clustering) thông tin đồng thuận và tách riêng thông tin mâu thuẫn.
    """

    def __init__(self):
        """Khởi tạo prompt builder."""
        pass

    def build_system_prompt(self) -> str:
        """
        Tạo lời nhắc hệ thống (System Instruction) tiêu chuẩn của một RAG đơn giản.
        
        Returns:
            str: Chỉ dẫn đóng vai trò trợ lý trả lời dựa trên tài liệu.
            
        TODO:
            - Thiết lập chỉ dẫn: "Bạn là trợ lý hỏi đáp. Hãy trả lời câu hỏi dựa trên ngữ cảnh được cung cấp."
        """
        return (
            "Bạn là một trợ lý thông minh. Dưới đây là các đoạn văn bản ngữ cảnh được truy xuất.\n"
            "Hãy trả lời câu hỏi của người dùng dựa vào ngữ cảnh này. Nếu không biết, hãy nói không biết."
        )
    
    def build_user_prompt(self, query: str, search_results: List[SearchResult]) -> str:
        """
        Ghép danh sách kết quả tìm kiếm và câu hỏi thành User Prompt hoàn chỉnh.
        
        Args:
            query: Câu hỏi của người dùng.
            search_results: Danh sách kết quả từ tầng Retrieval.
            
        Returns:
            str: Chuỗi văn bản prompt hoàn chỉnh.
            
        TODO:
            - Ghép các chunk.content thành chuỗi ngữ cảnh, đánh số thứ tự [Đoạn 1], [Đoạn 2]...
            - Thêm câu hỏi vào cuối prompt:
              "Ngữ cảnh:\n{context_text}\n\nCâu hỏi: {query}\nTrả lời:"
        """
        context_parts = [f"[Đoạn {i}] {sr.chunk.content}" for i, sr in enumerate(search_results, start=1)]
        context_text = "\n\n".join(context_parts)
        return f"Ngữ cảnh:\n{context_text}\n\nCâu hỏi: {query}\nTrả lời:"
