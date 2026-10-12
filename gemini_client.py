"""
Gemini LLM Client for GraphRAG Restaurant Recommendations.
Communicates directly with Google Gemini REST API with automatic model fallbacks,
retry backoff, and grounding against FoodLoversTV Knowledge Graph facts.
"""

import os
import json
import time
import logging
import urllib.request
import urllib.error
from typing import Dict, List, Optional, Any

logger = logging.getLogger("gemini_client")

CANDIDATE_MODELS = [
    "gemini-3.6-flash",
    "gemini-3.8-flash",
    "gemini-flash-latest",
]


class GeminiClient:
    """Client for Google Gemini REST API with fallback cascade and error handling."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: str = "gemini-3.6-flash",
        key_file: str = "gemini_api.txt"
    ):
        self.api_key = api_key or self._load_key(key_file)
        self.model_name = model_name
        self.key_file = key_file

    def _load_key(self, key_file: str) -> Optional[str]:
        """Load API key from Streamlit secrets, environment variable, or key file."""
        try:
            import streamlit as st
            if "GEMINI_API_KEY" in st.secrets:
                return str(st.secrets["GEMINI_API_KEY"]).strip()
        except Exception:
            pass

        if os.environ.get("GEMINI_API_KEY"):
            return os.environ["GEMINI_API_KEY"].strip()
        if os.path.exists(key_file):
            try:
                with open(key_file, "r", encoding="utf-8") as f:
                    key = f.read().strip()
                    if key:
                        return key
            except Exception as e:
                logger.warning(f"Could not read key from {key_file}: {e}")
        return None

    def set_api_key(self, key: str):
        """Update API key dynamically."""
        self.api_key = key.strip() if key else None

    def is_configured(self) -> bool:
        """Check if an API key is available."""
        return bool(self.api_key and len(self.api_key) > 10)

    def generate(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        temperature: float = 0.4,
        max_output_tokens: int = 2048,
    ) -> str:
        """Generate response from Gemini with automated model fallback and retries."""
        if not self.is_configured():
            raise ValueError(
                "Gemini API key not configured. Please enter a valid Gemini API key in the sidebar or check gemini_api.txt."
            )

        models_to_try = [self.model_name] + [m for m in CANDIDATE_MODELS if m != self.model_name]
        last_error = None

        payload: Dict[str, Any] = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {
                "temperature": temperature,
                "topP": 0.95,
                "topK": 40,
                "maxOutputTokens": max_output_tokens,
            },
        }
        if system_instruction:
            payload["system_instruction"] = {"parts": [{"text": system_instruction}]}

        for model in models_to_try:
            model_id = model if model.startswith("models/") else f"models/{model}"
            url = f"https://generativelanguage.googleapis.com/v1beta/{model_id}:generateContent?key={self.api_key}"

            for attempt in range(1, 3):
                try:
                    req = urllib.request.Request(
                        url,
                        data=json.dumps(payload).encode("utf-8"),
                        headers={"Content-Type": "application/json"},
                        method="POST"
                    )
                    with urllib.request.urlopen(req, timeout=30) as resp:
                        data = json.loads(resp.read().decode("utf-8"))
                        candidates = data.get("candidates", [])
                        if candidates:
                            parts = candidates[0].get("content", {}).get("parts", [])
                            if parts:
                                return parts[0].get("text", "")
                        return "I couldn't generate a recommendation from the graph at this time."

                except urllib.error.HTTPError as he:
                    error_body = ""
                    try:
                        error_body = he.read().decode("utf-8")
                    except Exception:
                        pass
                    last_error = f"HTTP {he.code}: {error_body or he.reason}"
                    logger.warning(f"Attempt with model {model} failed ({he.code}): {last_error}")

                    # If 503 or 429, wait briefly and retry or proceed to next model
                    if he.code in (503, 429) and attempt == 1:
                        time.sleep(1.5)
                        continue
                    break  # Try next candidate model

                except Exception as e:
                    last_error = str(e)
                    logger.warning(f"Error calling {model}: {e}")
                    break

        raise RuntimeError(f"All Gemini models failed. Last error: {last_error}")

    def answer_query_with_graphrag(
        self,
        query: str,
        llm_context: str,
        chat_history: Optional[List[Dict[str, str]]] = None,
    ) -> str:
        """
        Synthesize GraphRAG recommendation grounded in the retrieved Knowledge Graph.
        """
        system_instruction = (
            "You are the FoodLoversTV AI Concierge — an expert culinary guide inspired by Kripal Amanna's "
            "legendary food reviews across Bengaluru, Karnataka, and beyond.\n\n"
            "CRITICAL GUIDELINES:\n"
            "1. Ground your recommendations strictly on the provided Knowledge Graph Context.\n"
            "2. For each recommended restaurant:\n"
            "   - Clearly state the Restaurant Name, Neighborhood/Area, and Cuisine.\n"
            "   - Mention whether it is Veg or Non-veg and the Price Tier.\n"
            "   - Highlight signature Must-Try Dishes along with their prices and descriptions if present.\n"
            "   - Explain what makes the restaurant unique based on the FoodLoversTV review notes.\n"
            "   - Provide the direct YouTube Review Link in markdown format: [Watch FoodLoversTV Video](URL).\n"
            "3. If multiple options are retrieved, organize your answer with clear headers and bullet points.\n"
            "4. Maintain a warm, appetizing, and authoritative culinary tone.\n"
            "5. Do NOT invent restaurants or links not present in the context."
        )

        history_prompt = ""
        if chat_history:
            # Include last 2 conversational turns
            turns = chat_history[-4:]
            formatted_history = []
            for t in turns:
                role = "User" if t.get("role") == "user" else "Concierge"
                formatted_history.append(f"{role}: {t.get('content', '')}")
            history_prompt = "### PREVIOUS CONVERSATION:\n" + "\n".join(formatted_history) + "\n\n"

        full_prompt = (
            f"{history_prompt}"
            f"{llm_context}\n\n"
            f"### USER INQUIRY:\n{query}\n\n"
            f"Provide your culinary recommendation now:"
        )

        return self.generate(
            prompt=full_prompt,
            system_instruction=system_instruction,
            temperature=0.35,
        )
