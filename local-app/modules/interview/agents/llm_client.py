import logging
import json
import os
from typing import List, Dict, Any, Optional
from core.config import settings

logger = logging.getLogger(__name__)

class LLMClient:
    """
    Unified LLM Client wrapper supporting OpenAI, Gemini, Anthropic, and Groq APIs.
    Communicates via raw HTTP requests using 'requests' (with a fallback to 'urllib.request').
    """

    def __init__(self):
        try:
            import requests
            self.requests = requests
        except ImportError:
            self.requests = None
            logger.warning("requests library not found. Falling back to urllib.request.")

    def detect_provider(self) -> Optional[str]:
        """
        Determines the active provider based on environment variables or configured settings.
        Priority: Gemini -> OpenAI -> Anthropic -> Groq -> Ollama (Local model)
        """
        if os.getenv("KEIKO_OFFLINE_LLM") == "1":
            return None
        if os.getenv("GEMINI_API_KEY") or settings.GEMINI_API_KEY:
            return "gemini"
        if os.getenv("OPENAI_API_KEY") or settings.OPENAI_API_KEY:
            return "openai"
        if os.getenv("ANTHROPIC_API_KEY") or settings.ANTHROPIC_API_KEY:
            return "anthropic"
        if os.getenv("GROQ_API_KEY") or settings.GROQ_API_KEY:
            return "groq"
        
        # Check if local Ollama service is accessible on localhost:11434
        if self._check_ollama_available():
            return "ollama"

        return None

    def _check_ollama_available(self) -> bool:
        """Pings local Ollama server to check if a local LLM is available."""
        url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/tags"
        try:
            if self.requests:
                res = self.requests.get(url, timeout=0.3)
                if res.status_code == 200:
                    models = res.json().get("models", [])
                    return len(models) > 0
                return False
            else:
                import urllib.request
                req = urllib.request.Request(url, method="GET")
                with urllib.request.urlopen(req, timeout=0.3) as res:
                    if res.status == 200:
                        data = json.loads(res.read().decode("utf-8"))
                        return len(data.get("models", [])) > 0
                    return False
        except Exception:
            return False

    def complete(self, messages: List[Dict[str, str]], provider: Optional[str] = None, **kwargs) -> Optional[str]:
        """
        Submits chat messages to the target or auto-detected provider.
        """
        if not provider:
            provider = self.detect_provider()
            if not provider:
                logger.warning("No LLM API keys or local Ollama instance configured. Cannot complete chat.")
                return None

        provider = provider.lower()
        if provider == "openai":
            return self._call_openai(messages, **kwargs)
        elif provider == "gemini":
            return self._call_gemini(messages, **kwargs)
        elif provider == "anthropic":
            return self._call_anthropic(messages, **kwargs)
        elif provider == "groq":
            return self._call_groq(messages, **kwargs)
        elif provider == "ollama":
            return self._call_ollama(messages, **kwargs)
        else:
            logger.error(f"Unsupported provider: {provider}")
            return None

    def _call_ollama(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        url = f"{settings.OLLAMA_BASE_URL.rstrip('/')}/api/chat"
        headers = {"Content-Type": "application/json"}
        payload = {
            "model": kwargs.get("model", settings.OLLAMA_MODEL),
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.1)
            }
        }
        req_timeout = kwargs.get("timeout", 2.5)
        res_json = self._send_request(url, headers, payload, timeout=req_timeout)
        if res_json and "message" in res_json:
            return res_json["message"].get("content")
        elif res_json and "response" in res_json:
            return res_json.get("response")
        return None

    def _send_request(self, url: str, headers: Dict[str, str], payload: Dict[str, Any], timeout: float = 6.0) -> Optional[Dict[str, Any]]:
        """Sends HTTP POST request using either requests or urllib.request."""
        if self.requests:
            try:
                response = self.requests.post(url, headers=headers, json=payload, timeout=timeout)
                response.raise_for_status()
                return response.json()
            except Exception as e:
                logger.error(f"HTTP request failed using requests: {e}")
                if self.requests and hasattr(e, 'response') and e.response is not None:
                    logger.error(f"Response body: {e.response.text}")
                return None
        else:
            import urllib.request
            import urllib.error
            data_bytes = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(url, data=data_bytes, headers=headers, method="POST")
            try:
                with urllib.request.urlopen(req, timeout=timeout) as response:
                    res_bytes = response.read()
                    return json.loads(res_bytes.decode("utf-8"))
            except urllib.error.HTTPError as e:
                err_body = e.read().decode("utf-8") if e else ""
                logger.error(f"HTTPError in urllib: {e.code} - {e.reason}. Body: {err_body}")
                return None
            except Exception as e:
                logger.error(f"Exception in urllib: {e}")
                return None

    def _call_openai(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        api_key = os.getenv("OPENAI_API_KEY") or settings.OPENAI_API_KEY
        if not api_key:
            return None
        url = f"{settings.OPENAI_API_BASE.rstrip('/')}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": kwargs.get("model", settings.OPENAI_MODEL),
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 800)
        }
        req_timeout = kwargs.get("timeout", 6.0)
        res = self._send_request(url, headers, payload, timeout=req_timeout)
        if res and "choices" in res:
            try:
                return res["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as e:
                logger.error(f"Failed parsing OpenAI response choices: {e}")
        return None

    def _call_groq(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        api_key = os.getenv("GROQ_API_KEY") or settings.GROQ_API_KEY
        if not api_key:
            return None
        url = "https://api.groq.com/openai/v1/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": kwargs.get("model", settings.GROQ_MODEL),
            "messages": messages,
            "temperature": kwargs.get("temperature", 0.7),
            "max_tokens": kwargs.get("max_tokens", 800)
        }
        req_timeout = kwargs.get("timeout", 6.0)
        res = self._send_request(url, headers, payload, timeout=req_timeout)
        if res and "choices" in res:
            try:
                return res["choices"][0]["message"]["content"]
            except (KeyError, IndexError) as e:
                logger.error(f"Failed parsing Groq response choices: {e}")
        return None

    def _call_anthropic(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        api_key = os.getenv("ANTHROPIC_API_KEY") or settings.ANTHROPIC_API_KEY
        if not api_key:
            return None
        url = "https://api.anthropic.com/v1/messages"
        headers = {
            "x-api-key": api_key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json"
        }
        system_content = None
        filtered_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_content = msg["content"]
            else:
                filtered_messages.append(msg)

        payload = {
            "model": kwargs.get("model", settings.ANTHROPIC_MODEL),
            "messages": filtered_messages,
            "max_tokens": kwargs.get("max_tokens", 1024),
            "temperature": kwargs.get("temperature", 0.7)
        }
        if system_content:
            payload["system"] = system_content

        req_timeout = kwargs.get("timeout", 6.0)
        res = self._send_request(url, headers, payload, timeout=req_timeout)
        if res and "content" in res:
            try:
                return res["content"][0]["text"]
            except (KeyError, IndexError) as e:
                logger.error(f"Failed parsing Anthropic response content: {e}")
        return None

    def _call_gemini(self, messages: List[Dict[str, str]], **kwargs) -> Optional[str]:
        api_key = os.getenv("GEMINI_API_KEY") or settings.GEMINI_API_KEY
        if not api_key:
            logger.error("Gemini API key not found in environment or settings.")
            return None
        model = kwargs.get("model") or os.getenv("GEMINI_MODEL") or settings.GEMINI_MODEL or "gemini-1.5-flash"
        url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={api_key}"
        headers = {
            "Content-Type": "application/json"
        }
        contents = []
        system_instruction = None
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                system_instruction = {"parts": [{"text": content}]}
            else:
                contents.append({
                    "role": "user" if role == "user" else "model",
                    "parts": [{"text": content}]
                })

        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.7),
                "maxOutputTokens": kwargs.get("max_tokens", 800)
            }
        }
        if system_instruction:
            payload["systemInstruction"] = system_instruction

        req_timeout = kwargs.get("timeout", 6.0)
        res = self._send_request(url, headers, payload, timeout=req_timeout)
        if res and "candidates" in res:
            try:
                return res["candidates"][0]["content"]["parts"][0]["text"]
            except (KeyError, IndexError) as e:
                logger.error(f"Failed parsing Gemini response parts: {e}")
        return None
