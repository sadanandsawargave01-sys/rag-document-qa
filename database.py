from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from config import DB_CONFIG
from urllib.parse import quote_plus

db_user = DB_CONFIG.get("username") or DB_CONFIG.get("user")
db_password = quote_plus(DB_CONFIG["password"]) 

# ✅ PostgreSQL connection URL
connection_string = (
    f"postgresql+psycopg2://{db_user}:{db_password}"
    f"@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['database']}"
)

# ✅ Create engine
engine = create_engine(
    connection_string,
    pool_pre_ping=True,   # helps avoid stale connections
    pool_size=5,
    max_overflow=10
)

# ✅ Session factory
SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)

print("✅ Connected to PostgreSQL Database")

# ✅ Dependency (FastAPI style)
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
