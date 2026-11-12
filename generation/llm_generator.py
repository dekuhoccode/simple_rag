import os
import json
import time
import urllib.request
import urllib.error
from typing import List, Optional
from schema import SearchResult, GenerationOutput
from generation.prompt_builder import SimplePromptBuilder

class SimpleLLMGenerator:
    """
    Giai đoạn 5B: Generation - Sinh câu trả lời với LLM.
    
    Hỗ trợ linh hoạt 3 loại endpoint:
    1. Chuẩn OpenAI-Compatible REST API (Groq, DeepSeek, OpenAI, Ollama, vLLM...).
    2. Google Gemini AI Studio REST API (generativelanguage.googleapis.com).
    3. Google Vertex AI Gemini REST API (aiplatform.googleapis.com).
    
    Tích hợp Rate Limiter và cơ chế thử lại (Retry with Backoff) khi gặp HTTP 429 / 503.
    """

    # Danh sách domain nhận diện endpoint Gemini (AI Studio + Vertex AI)
    _GEMINI_DOMAINS = ("generativelanguage.googleapis.com", "aiplatform.googleapis.com")

    def __init__(
        self,
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        fallback_model_name: Optional[str] = None,
        temperature: float = 0.0,
        max_output_tokens: int = 512,
        max_rpm: int = 25,
        prompt_builder: Optional[SimplePromptBuilder] = None
    ):
        # 1. Base URL: Mặc định chuẩn OpenAI-compatible (Groq) nếu không chỉ định
        raw_base_url = (
            base_url
            or os.getenv("LLM_BASE_URL")
            or os.getenv("BASE_URL")
            or "https://api.groq.com/openai/v1"
        )
        self.base_url = raw_base_url.rstrip("/")

        # 2. API Key: Nếu endpoint là Gemini/Vertex AI, ưu tiên GEMINI_API_KEY
        if self._is_gemini_endpoint():
            self.api_key = (
                api_key
                or os.getenv("GEMINI_API_KEY")
                or os.getenv("LLM_API_KEY")
                or os.getenv("APIKEY", "")
            )
        else:
            self.api_key = (
                api_key
                or os.getenv("LLM_API_KEY")
                or os.getenv("APIKEY")
                or os.getenv("GEMINI_API_KEY", "")
            )

        # 3. Model Name & Fallback Model
        self.model_name = (
            model_name
            or os.getenv("LLM_MODEL")
            or os.getenv("MODEL")
            or "gemini-3.8-flash"
        )
        self.fallback_model_name = (
            fallback_model_name
            or os.getenv("FALLBACK_LLM_MODEL", self.model_name)
        )

        self.temperature = temperature
        self.max_output_tokens = max_output_tokens
        self.max_rpm = max_rpm
        self.prompt_builder = prompt_builder or SimplePromptBuilder()
        self._last_call_time = 0.0

    def _wait_for_rate_limit(self) -> None:
        """Đảm bảo khoảng cách an toàn giữa các request theo RPM."""
        if self.max_rpm <= 0:
            return
        min_interval = 60.0 / float(self.max_rpm)
        elapsed = time.time() - self._last_call_time
        if elapsed < min_interval:
            time.sleep(min_interval - elapsed)
        self._last_call_time = time.time()

    def _is_gemini_endpoint(self) -> bool:
        """Phát hiện endpoint Google Gemini (AI Studio hoặc Vertex AI)."""
        return any(domain in self.base_url for domain in self._GEMINI_DOMAINS)

    def _is_vertex_ai(self) -> bool:
        """Phát hiện endpoint Vertex AI."""
        return "aiplatform.googleapis.com" in self.base_url

    def _is_openai_compatible(self) -> bool:
        """Phát hiện endpoint chuẩn OpenAI chat completions (Groq, DeepSeek, Ollama...)."""
        return not self._is_gemini_endpoint()

    def _call_openai_compatible(self, system_prompt: str, user_prompt: str, model: str) -> str:
        """Gọi API chuẩn OpenAI (/chat/completions) cho Groq, DeepSeek, OpenAI, vLLM, Ollama."""
        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_output_tokens
        }

        data_bytes = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")

        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            choices = data.get("choices", [])
            if choices:
                return choices[0].get("message", {}).get("content", "").strip()
            return "Không nhận được phản hồi từ mô hình."

    def _call_gemini_api(self, system_prompt: str, user_prompt: str, model: str) -> str:
        """
        Gọi API Google Gemini REST (:generateContent).
        Hỗ trợ cả 2 loại endpoint:
        - Gemini AI Studio: https://generativelanguage.googleapis.com/v1beta
        - Vertex AI:        https://aiplatform.googleapis.com/v1/projects/.../models
        """
        # Xây dựng URL tùy theo loại endpoint
        if self._is_vertex_ai():
            # Vertex AI: Xử lý linh hoạt trường hợp base_url đã có sẵn model hoặc :generateContent
            clean_base = self.base_url.split(":generateContent")[0].rstrip("/")
            if clean_base.endswith(f"/{model}"):
                url = f"{clean_base}:generateContent?key={self.api_key}"
            elif "/models/" in clean_base:
                prefix = clean_base.split("/models")[0] + "/models"
                url = f"{prefix}/{model}:generateContent?key={self.api_key}"
            else:
                url = f"{clean_base}/{model}:generateContent?key={self.api_key}"
        else:
            # AI Studio: cần thêm tiền tố models/
            m_name = model if model.startswith("models/") else f"models/{model}"
            url = f"{self.base_url}/{m_name}:generateContent?key={self.api_key}"

        payload = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [{"role": "user", "parts": [{"text": user_prompt}]}],
            "generationConfig": {
                "temperature": self.temperature,
                "maxOutputTokens": self.max_output_tokens
            }
        }
        data_bytes = json.dumps(payload).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        }
        # Vertex AI cũng hỗ trợ Bearer token (OAuth2) — nếu key dạng ya29.* hoặc dài > 100 ký tự
        if self._is_vertex_ai() and (self.api_key.startswith("ya29.") or len(self.api_key) > 200):
            headers["Authorization"] = f"Bearer {self.api_key}"
            # Loại bỏ ?key= khỏi URL khi dùng Bearer
            url = url.split("?key=")[0]

        req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")

        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            candidates = data.get("candidates", [])
            if candidates:
                parts = candidates[0].get("content", {}).get("parts", [])
                return "".join(p.get("text", "") for p in parts).strip()
            return "Không nhận được phản hồi từ mô hình."

    def _call_api(self, system_prompt: str, user_prompt: str) -> str:
        """Gửi request tới LLM có hỗ trợ Rate Limiting và Fallback Model khi lỗi 429/503."""
        if not self.api_key or "YOUR_" in self.api_key:
            return f"[Mô phỏng LLM: Chưa cấu hình API Key] Phản hồi cho: {user_prompt[:60]}..."

        active_model = self.model_name
        max_retries = 3

        for attempt in range(max_retries):
            self._wait_for_rate_limit()
            try:
                if self._is_openai_compatible():
                    return self._call_openai_compatible(system_prompt, user_prompt, active_model)
                else:
                    return self._call_gemini_api(system_prompt, user_prompt, active_model)
            except urllib.error.HTTPError as e:
                err_text = e.read().decode("utf-8", errors="ignore")
                # Xử lý tự động chuyển fallback model khi gặp 404 (model_not_found), 429 (rate limit), 503 (quá tải)
                if e.code in (404, 429, 500, 502, 503) and attempt < max_retries - 1:
                    if self.fallback_model_name and active_model != self.fallback_model_name:
                        print(f"\n[Cảnh báo HTTP {e.code}] Model '{active_model}' gặp sự cố. Tự động chuyển sang model dự phòng: '{self.fallback_model_name}'...")
                        active_model = self.fallback_model_name
                        time.sleep(1.0)
                        continue
                    if e.code != 404:
                        time.sleep(2.0 * (attempt + 1))
                        continue
                print(f"[SimpleLLMGenerator HTTP {e.code}] {err_text[:200]}")
                return f"[Lỗi gọi LLM API ({e.code})]"
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(2.0)
                    continue
                print(f"[SimpleLLMGenerator Lỗi] {e}")
                return f"[Lỗi kết nối LLM: {e}]"

    def generate(self, query: str, search_results: List[SearchResult]) -> GenerationOutput:
        """Sinh câu trả lời Standard RAG: Prompt = System Prompt + Retrieved Contexts + Query."""
        system_prompt = self.prompt_builder.build_system_prompt()
        user_prompt = self.prompt_builder.build_user_prompt(query, search_results)

        answer_text = self._call_api(system_prompt, user_prompt)
        raw_prompt = f"{system_prompt}\n\n{user_prompt}"

        return GenerationOutput(
            query=query,
            answer=answer_text,
            retrieved_contexts=search_results,
            raw_prompt=raw_prompt
        )

    def generate_without_rag(self, query: str, system_prompt: Optional[str] = None) -> GenerationOutput:
        """Chế độ No-RAG: Sinh câu trả lời chỉ dựa vào trí nhớ nội tại của LLM."""
        sys_prompt = system_prompt or "You are a helpful and knowledgeable assistant. Answer directly and concisely."
        user_prompt = f"Question: {query}\nAnswer:"

        answer_text = self._call_api(sys_prompt, user_prompt)

        return GenerationOutput(
            query=query,
            answer=answer_text,
            retrieved_contexts=[],
            raw_prompt=f"{sys_prompt}\n\n{user_prompt}"
        )
