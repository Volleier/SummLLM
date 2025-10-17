import requests
from typing import Optional

class SimpleModelService:
    """前端模型服务：调用后端 FastAPI 服务；不可用时回退到临时摘要。"""

    def __init__(self, backend_url: Optional[str] = None, timeout: float = 10.0):
        # 默认后端地址（如有不同请传入）
        self.backend_url = backend_url or "http://localhost:8000"
        self.timeout = timeout

    def _health_ok(self) -> bool:
        try:
            r = requests.get(f"{self.backend_url}/health", timeout=2.0)
            if r.status_code == 200:
                data = r.json()
                return bool(data.get("model_loaded", False))
        except Exception:
            pass
        return False

    def summarize(self, text: str, max_length: int = 150, min_length: int = 30,
                  do_sample: bool = False, temperature: float = 0.7,
                  num_beams: int = 4, length_penalty: float = 2.0) -> str:
        """
        调用后端 /summarize 接口（POST），请求体与后端 SummaryRequest 对应。
        若后端不可用或请求失败，返回本地临时摘要。
        """
        # 参数映射到后端请求格式
        payload = {
            "text": text,
            "max_length": max_length,
            "min_length": min_length,
            "num_beams": num_beams,
            "length_penalty": length_penalty
        }

        # 优先检测后端并调用
        if self._health_ok():
            try:
                r = requests.post(f"{self.backend_url}/summarize", json=payload, timeout=self.timeout)
                if r.status_code == 200:
                    j = r.json()
                    # 兼容不同返回结构：SummaryResponse.summary 或 {'summary': ...}
                    if isinstance(j, dict):
                        if "summary" in j:
                            return j["summary"]
                        # 有些实现可能返回 {"summaries": [...]} 或 {"result": "..."}
                        if "summaries" in j and isinstance(j["summaries"], list) and j["summaries"]:
                            return j["summaries"][0]
                        if "result" in j:
                            return j["result"]
                    # 如果后端返回字符串直接返回
                    if isinstance(j, str):
                        return j
                else:
                    # 非 200，记录并回退
                    print(f"[model_service] 后端返回状态码 {r.status_code}, 回退到本地摘要。")
            except Exception as e:
                print(f"[model_service] 调用后端失败: {e}, 回退到本地摘要。")

        # 本地回退逻辑（保持现有占位行为）
        return self._local_fallback_summary(text, max_length, min_length, do_sample, temperature, num_beams)

    def _local_fallback_summary(self, text, max_length, min_length, do_sample, temperature, num_beams):
        # 简单的回退实现：取文本前若干字符并添加标记；可替换为更复杂的本地模型推理
        snippet = text.strip().replace("\n", " ")
        snippet = snippet[: max( min_length, min(200, max_length) )]
        return f"[LOCAL SUMMARY FALLBACK] {snippet}..."