import os
from typing import List
from schema import Document

class SimpleDocumentLoader:
    """
    Giai đoạn 1: Ingestion (Thu nạp dữ liệu).
    Lớp chịu trách nhiệm đọc tài liệu từ các nguồn khác nhau (file .txt, .md, chuỗi trực tiếp)
    và chuyển hóa thành các đối tượng Document chuẩn hóa.
    """

    def __init__(self):
        """Khởi tạo loader và các cấu hình liên quan (nếu có)."""
        pass

    def load_file(self, file_path: str) -> Document:
        """
        Đọc một file văn bản đơn lẻ (txt, md) từ đường dẫn.
        
        Args:
            file_path: Đường dẫn tuyệt đối hoặc tương đối tới file.
            
        Returns:
            Document: Đối tượng chứa doc_id, nội dung text đã làm sạch và metadata (tên file, kích thước).
            
        TODO: 
            - Mở file với encoding utf-8.
            - Gọi hàm clean_text để chuẩn hóa khoảng trắng thừa.
            - Trả về đối tượng Document.
        """
        with open(file_path, "r", encoding="utf-8") as f:
            raw_text = f.read()
        cleaned = self.clean_text(raw_text)
        return Document(
            doc_id=os.path.basename(file_path),
            content=cleaned,
            metadata={"source": file_path, "size": os.path.getsize(file_path)}
        )

    def load_directory(self, dir_path: str) -> List[Document]:
        """
        Duyệt qua một thư mục và đọc tất cả các file văn bản hợp lệ bên trong.
        
        Args:
            dir_path: Đường dẫn thư mục chứa tài liệu.
            
        Returns:
            List[Document]: Danh sách các tài liệu đã nạp.
            
        TODO:
            - Quét qua thư mục dir_path, tìm các file có đuôi .txt, .md.
            - Gọi self.load_file cho từng file và gom vào danh sách.
        """
        documents = []
        for filename in os.listdir(dir_path):
            if filename.endswith((".txt",".md")):
                full_path = os.path.join(dir_path, filename)
                documents.append(self.load_file(full_path))
        return documents

    def clean_text(self, text: str) -> str:
        """
        Làm sạch văn bản thô: loại bỏ ký tự lạ, chuẩn hóa khoảng trắng thừa, dấu xuống dòng liên tiếp.
        
        Args:
            text: Chuỗi văn bản thô ban đầu.
            
        Returns:
            str: Chuỗi văn bản đã được chuẩn hóa.
            
        TODO:
            - Thay thế các khoảng trắng kép bằng khoảng trắng đơn.
            - Xóa khoảng trắng ở đầu và cuối chuỗi.
        """
        import re
        text = re.sub(r'\s+', ' ', text)
        return text.strip()
