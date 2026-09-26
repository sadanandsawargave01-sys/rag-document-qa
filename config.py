import os


# -----------------------------
# Database Configuration
# -----------------------------
DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "database": os.getenv("DB_NAME", "rag_database"),
    "username": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", ""),
    "port": int(os.getenv("DB_PORT", "5432")),
}


# -----------------------------
# AWS / S3 Configuration
# -----------------------------
AWS_ACCESS_KEY = os.getenv("AWS_ACCESS_KEY")
AWS_SECRET_KEY = os.getenv("AWS_SECRET_KEY")
S3_BUCKET = os.getenv("S3_BUCKET")
S3_REGION = os.getenv("S3_REGION", "ap-south-1")


# -----------------------------
# Google / Gemini Configuration
# -----------------------------
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")


# -----------------------------
# Redis Configuration
# -----------------------------
REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))


# -----------------------------
# Optional Google Credentials
# -----------------------------
GOOGLE_CREDS = None


# -----------------------------
# Multilingual Configuration
# -----------------------------
SUPPORTED_LANGUAGES = ["en", "hi", "mr"]
DEFAULT_LANGUAGE = "en"


# -----------------------------
# RAGAS Configuration
# -----------------------------
RAGAS_EVALUATOR_MODEL = os.getenv(
    "RAGAS_EVALUATOR_MODEL",
    "gemini-2.5-flash"
)

RAGAS_EMBEDDING_MODEL = os.getenv(
    "RAGAS_EMBEDDING_MODEL",
    "models/gemini-embedding-001"
)
