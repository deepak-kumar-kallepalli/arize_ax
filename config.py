"""Shared Arize client + environment config.

Required env vars (put them in .env, see .env.example):
  ARIZE_API_KEY        - space API key (Settings -> API keys)
  ARIZE_SPACE_ID       - target space ID
  AI_INTEGRATION_ID    - AI integration used as the LLM judge
                         (Settings -> AI integrations)
"""

import os

from arize import ArizeClient
from dotenv import load_dotenv

load_dotenv(override=True)

SPACE_ID = os.environ["ARIZE_SPACE_ID"]
AI_INTEGRATION_ID = os.environ["AI_INTEGRATION_ID"]
JUDGE_MODEL_NAME = os.getenv("JUDGE_MODEL_NAME", "gpt-4o-mini")


def get_client() -> ArizeClient:
    return ArizeClient()  # reads ARIZE_API_KEY from env
