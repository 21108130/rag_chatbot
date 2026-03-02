

from __future__ import annotations

import os
from typing import Dict, List, Optional

from config.settings import settings
from src.utils.logger import logger
from src.utils.models import ConversationHistory, MessageRole

# Hardcoded working model — do not rely on settings for this
GROQ_MODEL = "llama-3.1-8b-instant"

RAG_CONTEXT_TEMPLATE = """Answer the question using ONLY the context below.
If the answer is not in the context, say "I could not find this in the uploaded documents."

Context:
{context}

Be concise. Cite [Source N] when referencing facts.
"""

NO_CONTEXT_PROMPT = "You are a helpful assistant. Ask the user to upload a document first."


class GroqLLMClient:

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> None:
        self._api_key = api_key
        self.model = GROQ_MODEL  # always use hardcoded working model
        self.temperature = 0.1
        self.max_tokens = 512
        self._client = None

    @property
    def api_key(self) -> str:
        if self._api_key:
            return self._api_key
        return os.environ.get("GROQ_API_KEY") or settings.groq_api_key

    @property
    def client(self):
        if self._client is None:
            key = self.api_key
            if not key:
                raise ValueError("GROQ_API_KEY is not set.")
            try:
                from groq import Groq
                self._client = Groq(api_key=key)
                logger.info(f"Groq client initialised (model={self.model})")
            except ImportError:
                raise ImportError("groq is not installed.")
        return self._client

    def generate(
        self,
        query: str,
        context: str = "",
        history: Optional[ConversationHistory] = None,
    ) -> Dict:
        messages = self._build_messages(query, context, history)
        logger.debug(f"[LLM] Sending {len(messages)} messages, model={self.model}")

        try:
            completion = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            answer = completion.choices[0].message.content.strip()
            tokens_used = completion.usage.total_tokens if completion.usage else None
            logger.info(f"[LLM] Response: {len(answer)} chars, tokens={tokens_used}")
            return {
                "answer": answer,
                "tokens_used": tokens_used,
                "model": self.model,
            }
        except Exception as e:
            error_msg = str(e)
            logger.error(f"[LLM] Error calling Groq API: {error_msg}")
         
            return {
                "answer": f"Sorry, I encountered an error: {error_msg}",
                "tokens_used": None,
                "model": self.model,
            }

    def _build_messages(
        self,
        query: str,
        context: str,
        history: Optional[ConversationHistory],
    ) -> List[Dict[str, str]]:

     
        context = (context or "")[:1500]

        if context.strip():
            system_content = RAG_CONTEXT_TEMPLATE.format(context=context)
        else:
            system_content = NO_CONTEXT_PROMPT

        messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_content}
        ]

        # Only last 2 turns
        if history:
            for msg in history.to_llm_messages(last_n=2):
                if msg["role"] != MessageRole.SYSTEM.value:
                    messages.append(msg)

      
        messages.append({"role": "user", "content": query[:300]})
        return messages

    def health_check(self) -> bool:
        try:
            result = self.generate(query="Say OK", context="")
            return True
        except Exception as exc:
            logger.warning(f"[LLM] Health check failed: {exc}")
            return False
