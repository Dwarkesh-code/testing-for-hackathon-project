# This file is deprecated.
# LLM calls are now handled directly via LangChain integrations:
#   - Groq  → langchain_groq.ChatGroq         (used in llm/router.py)
#   - NVIDIA NIM → langchain_nvidia_ai_endpoints.ChatNVIDIA (used in llm/extractor.py)
# Nothing in the pipeline imports from this file anymore.

class LLMClient:
    """Async wrapper around Groq and NVIDIA NIM OpenAI-compatible endpoints."""
    @staticmethod
    def _call_sync(endpoint: str, api_key: str, model: str, prompt: str, system_prompt: str) -> Optional[str]:
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}"
        }
        payload = {
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.2
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(endpoint, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=30) as res:
                response_json = json.loads(res.read().decode())
                return response_json["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as e:
            err_body = e.read().decode(errors="ignore")
            print(f"[LLM HTTP Error {e.code}] Model {model}: {err_body}")
            return None
        except Exception as e:
            print(f"[LLM Error] {str(e)}")
            return None

    @classmethod
    async def call_json(cls, prompt: str, system_prompt: str, use_extractor: bool = False) -> Optional[Dict[str, Any]]:
        # First try Groq
        groq_key = config.GROQ_KEYS.get_key()
        if groq_key:
            model = config.EXTRACTOR_MODEL if use_extractor else config.ROUTER_MODEL
            res = await asyncio.to_thread(
                cls._call_sync,
                "https://api.groq.com/openai/v1/chat/completions",
                groq_key,
                model,
                prompt,
                system_prompt
            )
            if res:
                try:
                    return json.loads(res)
                except Exception:
                    pass

        # Fallback to NVIDIA NIM
        nvidia_key = config.NVIDIA_KEYS.get_key()
        if nvidia_key:
            model = "meta/llama-3.3-70b-instruct" if use_extractor else "meta/llama-3.1-8b-instruct"
            res = await asyncio.to_thread(
                cls._call_sync,
                "https://integrate.api.nvidia.com/v1/chat/completions",
                nvidia_key,
                model,
                prompt,
                system_prompt
            )
            if res:
                try:
                    return json.loads(res)
                except Exception:
                    pass

        return None
