"""This module provides a shared async database engine for all other modules to import."""

import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from dotenv import load_dotenv

load_dotenv()

DATABASE_URL = (
    f"postgresql+asyncpg://"
    f"{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
    f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}"
    f"/{os.getenv('POSTGRES_DB')}"
)

engine = create_async_engine(
    DATABASE_URL,
    pool_size = 10, # Max 10 persistent connections
    max_overflow = 20, # Up to 20 extra connections under load
    pool_pre_ping = True, # Test connections before using them
    echo=False, # Set True to log all SQL — useful for debugging
)

AsyncSessionLocal = sessionmaker(
    bind=engine, # Attach all sessions to this engine.
    class_=AsyncSession, # Create async sessions, not regular sessions.
    expire_on_commit=False # Keep objects accessible after commit
)

async def get_db():
    """Dependency-injection style session getter. Use as an async context manager."""
    async with AsyncSessionLocal() as session:
        try:
            yield session # Hands the session to whoever called get_db()
            await session.commit()
        except Exception:
            await session.rollback() # Hands the session to whoever called get_db()
            raise
        finally:
            await session.close()