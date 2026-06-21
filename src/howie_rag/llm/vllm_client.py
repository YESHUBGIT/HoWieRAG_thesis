import json
from typing import Optional
from urllib import request

from howie_rag.llm.base import BaseLLMClient


class VLLMClient(BaseLLMClient):
    def __init__(self, base_url: str, model_name: str, timeout: int = 120) -> None:
        self.base_url = base_url.rstrip("/")
        self.model_name = model_name
        self.timeout = timeout

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        max_tokens: int = 400,
    ) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response_data = self._post_json("/v1/chat/completions", payload)
        return response_data["choices"][0]["message"]["content"].strip()

    def _post_json(self, path: str, payload: dict) -> dict:
        http_request = request.Request(
            url=self.base_url + path,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with request.urlopen(http_request, timeout=self.timeout) as response:
            body = response.read().decode("utf-8")
        return json.loads(body)
