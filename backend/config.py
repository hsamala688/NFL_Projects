import os
from pathlib import Path
from dotenv import load_dotenv

# Load backend-specific overrides first, then fall back to root .env.
# backend/.env is gitignored and carries FRONTEND_ORIGIN and any
# backend-specific settings. The GOOGLE_API_KEY lives in root .env.
_backend_env = Path(__file__).parent / ".env"
if _backend_env.exists():
    load_dotenv(_backend_env)
else:
    load_dotenv(Path(__file__).parent.parent / ".env")

FRONTEND_ORIGIN: str = os.getenv("FRONTEND_ORIGIN", "http://localhost:5173")
MAX_QUESTION_LENGTH: int = 500
# Set in backend/.env. If empty the live /ask endpoint rejects all requests.
AGENT_API_KEY: str = os.getenv("AGENT_API_KEY", "")
