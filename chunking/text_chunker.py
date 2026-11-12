from typing import List
from schema import Document, Chunk, SourceType

class SimpleTextChunker:
    """
    Giai đoạn 2: Chunking (Cắt đoạn văn bản).
    
    Trong mô hình Simple/Naive RAG, văn bản thường được cắt theo kích thước cố định (fixed-size chunking)
    với một khoảng gối đầu (overlap) trượt qua chuỗi ký tự.
    
    * LƯU Ý HỌC TẬP & HẠN CHẾ SO VỚI BÀI BÁO ASTUTE RAG:
      Cách cắt thô này thường làm:
      1. Đứt gãy câu giữa chừng hoặc tách rời câu khỏi tiêu đề cha (heading context).
      2. Khi truy xuất, các chunk này có thể chứa thông tin không hoàn chỉnh, mơ hồ hoặc nhiễu,
         dẫn đến hiện tượng "Imperfect Retrieval" mà bài báo ASTUTE RAG chỉ ra là nguyên nhân
         hàng đầu gây ra xung đột tri thức (Knowledge Conflicts).
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Khởi tạo chunker với kích thước mục tiêu và độ dài gối đầu.
        
        Args:
            chunk_size: Số lượng ký tự tối đa của một chunk.
            chunk_overlap: Số lượng ký tự lặp lại giữa chunk trước và chunk sau để giữ tính liên tục.
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split_text(self, text: str) -> List[str]:
        """
        Cắt chuỗi văn bản dài thành danh sách các chuỗi con bằng thuật toán cửa sổ trượt (sliding window).
        
        Args:
            text: Nội dung văn bản cần chia nhỏ.
            
        Returns:
            List[str]: Danh sách các đoạn văn bản (text pieces).
            
        TODO:
            - Dùng vòng lặp với bước nhảy: step = chunk_size - chunk_overlap.
            - Cắt từng đoạn: text[start : start + chunk_size].
            - Xử lý điều kiện dừng khi đến cuối văn bản.
        """
        splits = []
        start = 0
        step = self.chunk_size - self.chunk_overlap
        if step <= 0:
            step = self.chunk_size

        while start < len(text):
            end = start + self.chunk_size
            splits.append(text[start:end])
            start += step

        return splits

    def chunk_document(self, document: Document) -> List[Chunk]:
        """
        Nhận vào 1 đối tượng Document, cắt nhỏ nội dung và đóng gói thành danh sách các Chunk có ID duy nhất.
        
        Args:
            document: Đối tượng Document cần cắt.
            
        Returns:
            List[Chunk]: Danh sách các đối tượng Chunk với chunk_id theo định dạng {doc_id}_c{index}.
            
        TODO:
            - Gọi split_text(document.content).
            - Tạo đối tượng Chunk cho mỗi mảnh văn bản với metadata và source_type=EXTERNAL.
        """
        text_pieces = self.split_text(document.content)
        chunks = []
        for i, piece in enumerate(text_pieces):
            chunk = Chunk(
                chunk_id=f"{document.doc_id}_c{i}",
                doc_id=document.doc_id,
                content=piece,
                source_type=SourceType.EXTERNAL,
                metadata={"chunk_index":i}
            )
            chunks.append(chunk)
        return chunks
        
    def chunk_documents(self, documents: List[Document]) -> List[Chunk]:
        """
        Thực hiện cắt hàng loạt danh sách các Document.
        
        Args:
            documents: Danh sách Document.
            
        Returns:
            List[Chunk]: Toàn bộ các Chunk được tạo ra từ tất cả tài liệu.
            
        TODO:
            - Duyệt qua từng Document trong danh sách.
            - Gọi chunk_document và tổng hợp kết quả vào một danh sách duy nhất.
        """
        all_chunks = []
        for doc in documents:
            all_chunks.extend(self.chunk_document(doc))
        return all_chunks
