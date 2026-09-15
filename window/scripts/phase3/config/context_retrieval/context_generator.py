import os
import re
import time
import logging
import warnings
from typing import Optional

warnings.filterwarnings("ignore", category=FutureWarning)
import google.generativeai as genai
from dotenv import load_dotenv


from scripts.phase3.config.context_retrieval.context_cache import ContextCache

load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"))

logger = logging.getLogger(__name__)

PROMPT_VERSION = "v1"  # bump this when you change the prompt
MODEL_NAME = "models/gemini-3.6-flash"


PROMPT_TEMPLATE = """You are helping improve document retrieval.

<document>
{document}
</document>

Here is the chunk we want to situate within the document:

<chunk>
{chunk}
</chunk>

Give a short, self-contained context (1-2 sentences, max 80 words) that explains:
- Where this chunk fits in the document (section, topic).
- What entity, rule, or directive it is about.
- Any key identifier (circular number, section reference, dates).

Respond with ONLY the context sentences. No preamble, no bullet points, no quotes.
"""


def _extract_text(response) -> str:
    """Robustly extract text from Gemini response."""
    try:
        text = response.text
    except Exception:
        text = ""
        for cand in getattr(response, "candidates", []) or []:
            for part in getattr(cand.content, "parts", []) or []:
                text += getattr(part, "text", "")
    return text.strip()


class ContextGenerator:
    def __init__(self, cache: Optional[ContextCache] = None):
        self.cache = cache or ContextCache()
        self.model = genai.GenerativeModel(MODEL_NAME)
        self.calls_made = 0
        self.cache_hits = 0

    def generate_context(
        self,
        doc_id: str,
        chunk_index: int,
        chunk_text: str,
        document_text: str,
        retries: int = 3,
    ) -> str:
        key = self.cache.make_key(doc_id, chunk_index, chunk_text, PROMPT_VERSION)
        cached = self.cache.get(key)
        if cached is not None:
            self.cache_hits += 1
            return cached

        # Truncate document to a safe size for the prompt
        doc_snippet = document_text[:40_000]  # ~10k tokens
        prompt = PROMPT_TEMPLATE.format(document=doc_snippet, chunk=chunk_text)

        last_err = None
        for attempt in range(retries):
            try:
                response = self.model.generate_content(
                    prompt,
                    generation_config={"temperature": 0.2, "max_output_tokens": 120},
                )
                context = _extract_text(response)
                context = re.sub(r"\s+", " ", context).strip()
                if not context:
                    raise ValueError("Empty response")

                self.cache.set(key, context)
                self.calls_made += 1
                return context
            except Exception as e:
                last_err = e
                wait = 2 ** attempt
                logger.warning(f"Gemini call failed (attempt {attempt+1}): {e}. Retrying in {wait}s")
                time.sleep(wait)

        logger.error(f"All retries failed for chunk {chunk_index} of {doc_id}: {last_err}")
        return ""  # Empty context — chunk stays raw

    def stats(self) -> dict:
        return {
            "api_calls": self.calls_made,
            "cache_hits": self.cache_hits,
            **self.cache.stats(),
        }