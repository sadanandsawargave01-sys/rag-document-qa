
from sqlalchemy import Column, Integer, String, DateTime, func, Boolean
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class QABOTParser(Base):
    __tablename__ = "qabot_parser"   # ✅ Postgres table name (lowercase)

    id = Column(Integer, primary_key=True, index=True, autoincrement=True)

    client_id = Column(Integer, nullable=False)
    user_id = Column(Integer, nullable=False)

    category_id = Column(String(50), nullable=False)

    file_uuid = Column(String(50), nullable=False)
    file_name = Column(String(255), nullable=False)

    status = Column(String(50), nullable=False)

    s3_location = Column(String(255), nullable=True)

    created_by = Column(Integer, nullable=False)

    # ✅ PostgreSQL timestamp (replaces getdate())
    created_on = Column(DateTime(timezone=True), server_default=func.now())

    file_type = Column(String(50), nullable=False)
    file_size = Column(Integer, nullable=False)

    index_name = Column(String(255), nullable=False)

    pages = Column(Integer, nullable=True)

    # ✅ Additional fields (based on your new schema)
    is_favourite = Column(Boolean, default=False)
    favourite_on = Column(DateTime(timezone=True), nullable=True)
