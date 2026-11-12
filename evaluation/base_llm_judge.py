import os
import re
import time
import json
import random
import urllib.request
import urllib.error
from typing import Optional, Dict, Any
from config import SimpleRAGConfig


class SharedRateLimiter:
    """
    Bộ điều tiết nhịp độ gọi API toàn cục (Rate Limiter).
    Đảm bảo khoảng cách giữa 2 request liên tiếp luôn >= 60 / max_rpm để tránh lỗi 429.
    """
    _last_request_time: float = 0.0

    @classmethod
    def wait_for_slot(cls, max_rpm: int) -> None:
        if max_rpm <= 0:
            return
        min_interval = 60.0 / float(max_rpm)
        elapsed = time.time() - cls._last_request_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        cls._last_request_time = time.time()


class BaseLLMJudge:
    """
    Lớp cơ sở cho các bộ chấm điểm LLM-as-a-Judge (Dựa trên kiến trúc của Custom-RAG).
    - Hỗ trợ linh hoạt 3 loại endpoint:
      1. Chuẩn OpenAI-Compatible REST API (Groq, DeepSeek, OpenAI, vLLM, Ollama).
      2. Google Gemini AI Studio REST API (generativelanguage.googleapis.com).
      3. Google Vertex AI Gemini REST API (aiplatform.googleapis.com).
    - Tự động bắt lỗi HTTP 429/503 và thử lại (Retry with exponential backoff).
    - Tự động trích xuất chuỗi JSON phản hồi an toàn.
    - Có chế độ Fallback mô phỏng nếu người dùng chưa điền API Key, giúp pipeline không bị crash.
    """

    # Danh sách domain nhận diện endpoint Gemini (AI Studio + Vertex AI)
    _GEMINI_DOMAINS = ("generativelanguage.googleapis.com", "aiplatform.googleapis.com")

    def __init__(self, config: Optional[SimpleRAGConfig] = None):
        self.config = config or SimpleRAGConfig()
        
        self.gemini_key = getattr(self.config, "gemini_api_key", "") or os.getenv("GEMINI_API_KEY", "")
        self.llm_key = getattr(self.config, "llm_api_key", "") or os.getenv("LLM_API_KEY", "") or os.getenv("APIKEY", "")
        
        raw_base_url = (
            getattr(self.config, "judge_base_url", None)
            or os.getenv("JUDGE_BASE_URL")
            or getattr(self.config, "llm_base_url", None)
            or os.getenv("LLM_BASE_URL")
            or os.getenv("BASE_URL")
            or "https://api.groq.com/openai/v1"
        )
        self.base_url = raw_base_url.rstrip("/")
        
        if self._is_gemini_endpoint():
            # Endpoint Gemini (AI Studio hoặc Vertex AI): ưu tiên GEMINI_API_KEY
            self.api_key = self.gemini_key or self.llm_key
            self.model_name = getattr(self.config, "judge_model", "gemini-3.8-flash")
            # AI Studio cần tiền tố models/, Vertex AI thì không (vì base_url đã chứa .../models)
            if not self._is_vertex_ai() and not self.model_name.startswith("models/"):
                self.model_name = f"models/{self.model_name}"
        else:
            # OpenAI-compatible (Groq, DeepSeek, Ollama...)
            self.api_key = self.llm_key or self.gemini_key
            self.model_name = (
                getattr(self.config, "judge_model", None)
                or getattr(self.config, "llm_model", "llama-3.1-8b-instant")
            )
            
        self.max_rpm = getattr(self.config, "judge_max_rpm", 20)
        self.temperature = getattr(self.config, "judge_temperature", 0.0)

    @property
    def has_valid_api_key(self) -> bool:
        """Kiểm tra xem API Key đã được cấu hình hợp lệ chưa."""
        return bool(self.api_key and "YOUR_" not in self.api_key and len(self.api_key.strip()) > 10)

    def _is_gemini_endpoint(self) -> bool:
        """Phát hiện endpoint Google Gemini (AI Studio hoặc Vertex AI)."""
        return any(domain in self.base_url for domain in self._GEMINI_DOMAINS)

    def _is_vertex_ai(self) -> bool:
        """Phát hiện endpoint Vertex AI."""
        return "aiplatform.googleapis.com" in self.base_url

    def _is_openai_compatible(self) -> bool:
        """Phát hiện endpoint chuẩn OpenAI (Groq, DeepSeek, Ollama...)."""
        return not self._is_gemini_endpoint()

    def _call_api(self, system_prompt: str, user_prompt: str, max_retries: int = 3) -> Dict[str, Any]:
        """
        Gửi yêu cầu chấm điểm tới LLM (Groq / OpenAI / Gemini / Vertex AI) và nhận phản hồi dạng JSON.
        """
        if not self.has_valid_api_key:
            return {}

        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }

        if self._is_openai_compatible():
            # --- Endpoint chuẩn OpenAI-compatible (Groq, DeepSeek...) ---
            url = f"{self.base_url}/chat/completions"
            headers["Authorization"] = f"Bearer {self.api_key}"
            payload = {
                "model": self.model_name,
                "messages": [
                    {"role": "system", "content": system_prompt + "\nCHÚ Ý: Chỉ trả về duy nhất chuỗi JSON hợp lệ, không kèm văn bản hay định dạng markdown bên ngoài."},
                    {"role": "user", "content": user_prompt}
                ],
                "temperature": self.temperature,
                "response_format": {"type": "json_object"}
            }
        else:
            # --- Endpoint Google Gemini (AI Studio hoặc Vertex AI) ---
            if self._is_vertex_ai():
                clean_base = self.base_url.split(":generateContent")[0].rstrip("/")
                if clean_base.endswith(f"/{self.model_name}"):
                    url = f"{clean_base}:generateContent?key={self.api_key}"
                elif "/models/" in clean_base:
                    prefix = clean_base.split("/models")[0] + "/models"
                    url = f"{prefix}/{self.model_name}:generateContent?key={self.api_key}"
                else:
                    url = f"{clean_base}/{self.model_name}:generateContent?key={self.api_key}"
            else:
                url = f"{self.base_url}/{self.model_name}:generateContent?key={self.api_key}"

            payload = {
                "systemInstruction": {"parts": [{"text": system_prompt}]},
                "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
                "generationConfig": {
                    "temperature": self.temperature,
                    "responseMimeType": "application/json"
                }
            }

            # Vertex AI: hỗ trợ Bearer token (OAuth2) nếu key dạng ya29.*
            if self._is_vertex_ai() and (self.api_key.startswith("ya29.") or len(self.api_key) > 200):
                headers["Authorization"] = f"Bearer {self.api_key}"
                url = url.split("?key=")[0]

        json_bytes = json.dumps(payload).encode("utf-8")

        for attempt in range(max_retries):
            SharedRateLimiter.wait_for_slot(self.max_rpm)
            req = urllib.request.Request(
                url,
                data=json_bytes,
                headers=headers,
                method="POST"
            )

            try:
                with urllib.request.urlopen(req, timeout=60) as resp:
                    resp_bytes = resp.read()
                    data = json.loads(resp_bytes.decode("utf-8"))
                    if self._is_openai_compatible():
                        choices = data.get("choices", [])
                        raw_content = choices[0].get("message", {}).get("content", "") if choices else ""
                        return self._extract_json_from_text(raw_content)
                    else:
                        return self._extract_json_safely(data)
            except urllib.error.HTTPError as e:
                err_code = e.code
                err_body = e.read().decode("utf-8", errors="ignore")
                
                if err_code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    wait_time = (10.0 * (attempt + 1)) + random.uniform(0.5, 2.0)
                    print(f"[{self.__class__.__name__} HTTP {err_code}] Quá tải/giới hạn nhịp, đợi {wait_time:.1f}s...")
                    time.sleep(wait_time)
                    continue
                print(f"[{self.__class__.__name__}] Lỗi HTTP {err_code}: {err_body[:200]}")
                break
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(3.0 * (attempt + 1))
                    continue
                print(f"[{self.__class__.__name__}] Lỗi kết nối API: {e}")
                break

        return {}

    @staticmethod
    def _extract_json_from_text(raw_text: str) -> dict:
        """Trích xuất chuỗi JSON từ văn bản."""
        if not raw_text or not isinstance(raw_text, str):
            return {}
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_text.strip(), flags=re.IGNORECASE).strip()
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
            return {}

    @staticmethod
    def _extract_json_safely(api_data: dict) -> dict:
        """Trích xuất và làm sạch chuỗi JSON từ kết quả trả về của Gemini."""
        if not api_data or not isinstance(api_data, dict):
            return {}
        candidates = api_data.get("candidates", [])
        if not candidates:
            return {}
        parts = candidates[0].get("content", {}).get("parts", [])
        raw_text = "".join(p.get("text", "") for p in parts).strip()
        return BaseLLMJudge._extract_json_from_text(raw_text)
