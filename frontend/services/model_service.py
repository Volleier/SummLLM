import requests
from typing import Optional

class SimpleModelService:
    """Frontend model service: calls backend FastAPI service, falls back to a default message on failure."""

    def __init__(self, backend_url: Optional[str] = None, timeout: float = 10.0):
        # backend URL and timeout
        self.backend_url = backend_url or "http://localhost:8000"
        self.timeout = timeout

    def _health_ok(self) -> bool:
        # simple health check: ensure model is loaded
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
        """Call backend /summarize, fall back locally on failure."""
        payload = {
            "text": text,
            "max_length": max_length,
            "min_length": min_length,
            "num_beams": num_beams,
            "length_penalty": length_penalty
        }

        if self._health_ok():
            try:
                r = requests.post(f"{self.backend_url}/summarize", json=payload, timeout=self.timeout)
                if r.status_code == 200:
                    j = r.json()
                    if isinstance(j, dict):
                        if "summary" in j:
                            return j["summary"]
                        if "summaries" in j and isinstance(j["summaries"], list) and j["summaries"]:
                            return j["summaries"][0]
                        if "result" in j:
                            return j["result"]
                    if isinstance(j, str):
                        return j
                else:
                    print(f"[model_service] Backend returned status code {r.status_code}, falling back to local summary.")
            except Exception as e:
                print(f"[model_service] Failed to call backend: {e}, falling back to local summary.")

        return self._local_fallback_summary(text, max_length, min_length, do_sample, temperature, num_beams)

    def _local_fallback_summary(self, text, max_length, min_length, do_sample, temperature, num_beams):
        # local fallback: return a fixed default message
        return "[LOCAL SUMMARY FALLBACK] Sorry, unable to generate a summary. Please try again later."
