"""Shared paths and prompt for the local English-to-SQL experiment."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
REVISION = "7ae557604adf67be50417f59c2c2f167def9a775"
MODEL_DIR = PROJECT_ROOT / "models" / "qwen2.5-0.5b-instruct"
TRAIN_PATH = PROJECT_ROOT / "data" / "train.jsonl"
CPU_THREADS = 4
MAX_NEW_TOKENS = 64

SYSTEM_PROMPT = """Translate the user's request into SQLite SQL.
Return only one read-only SELECT statement.

Database schema:
{schema}"""
