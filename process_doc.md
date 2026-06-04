# Douban NLP Observatory — Complete Process Document

> **Purpose:** A self-contained, step-by-step guide for building a Chinese NLP observatory that ingests live Douban group threads, processes them through a full NLP pipeline, and visualizes semantic change over time. Designed to be followed independently, covering async scraping, production-grade storage, Chinese text preprocessing, fine-tuning Chinese language models, and time-series visualization. This is not a toy project — the stack is production-scale from day one.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Environment Setup](#2-environment-setup)
3. [Phase 1 — Infrastructure: Docker, Postgres/TimescaleDB, Qdrant](#3-phase-1--infrastructure-docker-postgrestimescaledb-qdrant)
4. [Phase 2 — The Douban Scraping Client](#4-phase-2--the-douban-scraping-client)
5. [Phase 3 — Storage Schema Design](#5-phase-3--storage-schema-design)
6. [Phase 4 — Chinese Text Preprocessing Pipeline](#6-phase-4--chinese-text-preprocessing-pipeline)
7. [Phase 5 — Keyword Extraction & Lexical Analysis](#7-phase-5--keyword-extraction--lexical-analysis)
8. [Phase 6 — Sentiment Analysis with Pretrained Chinese Models](#8-phase-6--sentiment-analysis-with-pretrained-chinese-models)
9. [Phase 7 — Topic Modeling with BERTopic](#9-phase-7--topic-modeling-with-bertopic)
10. [Phase 8 — Fine-Tuning a Chinese BERT Model](#10-phase-8--fine-tuning-a-chinese-bert-model)
11. [Phase 9 — LoRA Fine-Tuning for Thread Summarization](#11-phase-9--lora-fine-tuning-for-thread-summarization)
12. [Phase 10 — Understanding What the Model Learned (Probing)](#12-phase-10--understanding-what-the-model-learned-probing)
13. [Phase 11 — The 5-Minute Orchestration Pipeline](#13-phase-11--the-5-minute-orchestration-pipeline)
14. [Phase 12 — Streamlit Time-Series Dashboard](#14-phase-12--streamlit-time-series-dashboard)
15. [Phase 13 — Tokenizer Training on Your Own Corpus](#15-phase-13--tokenizer-training-on-your-own-corpus)
16. [Phase 14 — Model Distillation](#16-phase-14--model-distillation)
17. [Reference: Key Concepts Glossary](#17-reference-key-concepts-glossary)
18. [Reference: Common Errors & Fixes](#18-reference-common-errors--fixes)

---

## 1. Project Overview

### What You Are Building

A live Chinese NLP observatory that:
- Scrapes Douban group threads every 5 minutes via an async Python client
- Stores raw text and NLP results in a production-grade time-indexed database
- Runs a batched NLP pipeline after every scrape cycle: keyword extraction, sentiment analysis, topic modeling, and thread summarization
- Visualizes how language, sentiment, and topics evolve over time — per minute, hour, day, or month — through a Streamlit dashboard with lexical clouds and time-series charts
- Includes fine-tuned Chinese BERT models trained on your own scraped data

### How It Works (Bird's Eye View)

```
Douban Group Pages (HTML)
        ↓
  Async DoubanClient (httpx + Playwright)
  Rate-limited, session-managed, resumable
        ↓
  PostgreSQL + TimescaleDB
  Raw threads & replies, time-indexed hypertables
        ↓
  NLP Pipeline (triggered every 5 min after scrape)
  ├── jieba segmentation + stopword filtering
  ├── TF-IDF + TextRank keyword extraction
  ├── MacBERT-based sentiment classifier
  ├── BERTopic topic modeling
  └── Qwen2 LoRA summarizer
        ↓
  Pre-computed aggregates written back to TimescaleDB
  Vector embeddings written to Qdrant
        ↓
  Streamlit Dashboard
  Time-series keyword trends, sentiment trajectory,
  topic heatmap, lexical clouds, thread browser
```

### Skills You Will Learn

- Async Python scraping with rate limiting, session management, and anti-detection patterns
- Production database schema design for time-series NLP data
- Chinese-specific text preprocessing (why it is fundamentally different from English)
- Keyword extraction with TF-IDF and graph-based TextRank
- Zero-shot and fine-tuned sentiment analysis with Chinese BERT models
- Neural topic modeling with BERTopic on Chinese text
- Full fine-tuning of MacBERT on a domain-specific classification task
- LoRA parameter-efficient fine-tuning on a 1.5B Chinese language model
- Probing classifiers — understanding what each transformer layer encodes
- Tokenizer training from scratch on a domain corpus
- Knowledge distillation — compressing a fine-tuned model for fast inference
- APScheduler-based pipeline orchestration with strict 5-minute cycle management
- Streamlit dashboards with Plotly time-series charts and Chinese wordclouds

### Final File Structure

```
douban-observatory/
├── docker-compose.yml              # Postgres/TimescaleDB + Qdrant + app
├── .env                            # Secrets and config — never commit
├── .gitignore
├── requirements.txt
├── notebooks/
│   ├── 01_scraper_exploration.ipynb
│   ├── 02_schema_validation.ipynb
│   ├── 03_preprocessing_exploration.ipynb
│   ├── 04_keyword_exploration.ipynb
│   ├── 05_sentiment_baseline.ipynb
│   ├── 06_bertopic_exploration.ipynb
│   ├── 07_finetuning_prep.ipynb
│   ├── 08_probing_experiments.ipynb
│   └── 09_distillation_eval.ipynb
├── src/
│   ├── client/
│   │   ├── douban_client.py        # DoubanClient — the scraping abstraction
│   │   └── session_manager.py      # Cookie/session handling
│   ├── db/
│   │   ├── connection.py           # Async SQLAlchemy engine
│   │   ├── models.py               # ORM models
│   │   └── migrations/
│   │       └── 001_initial.sql     # Schema definition
│   ├── pipeline/
│   │   ├── preprocessor.py         # Chinese text cleaning + segmentation
│   │   ├── keywords.py             # TF-IDF + TextRank keyword extraction
│   │   ├── sentiment.py            # Sentiment classifier (pretrained + fine-tuned)
│   │   ├── topics.py               # BERTopic wrapper
│   │   ├── summarizer.py           # LoRA summarizer wrapper
│   │   └── aggregator.py           # Writes pre-computed aggregates
│   ├── training/
│   │   ├── finetune_bert.py        # Full fine-tuning script for MacBERT
│   │   ├── finetune_lora.py        # LoRA fine-tuning script for Qwen2
│   │   ├── probing.py              # Probing classifier experiments
│   │   ├── distill.py              # Distillation training script
│   │   └── tokenizer_train.py      # Custom tokenizer training
│   ├── scheduler.py                # APScheduler orchestration
│   └── utils.py                    # Shared helpers
├── app.py                          # Streamlit entry point
├── models/                         # Saved model checkpoints (gitignored)
└── data/
    ├── stopwords_zh.txt            # Chinese stopword list
    └── raw/                        # Raw HTML snapshots for debugging
```

---

## 2. Environment Setup

### 2.1 Prerequisites

- Python 3.10 or higher
- Docker Desktop (free) — required for Postgres, TimescaleDB, and Qdrant
- A Douban account (free) — needed for login-gated group content
- Git
- A Kaggle account — for accessing GPU compute in later phases
- 8GB+ RAM recommended; 16GB preferred for fine-tuning phases locally

### 2.2 Create and Activate a Virtual Environment

Open Ubuntu first (if you don't have it, do `wsl --install`)

```bash
# Inside Ubuntu (WSL terminal)
sudo apt update
sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv python3-full build-essential

# Create and activate venv
python -m venv venv
source venv/bin/activate          # Mac/Linux
venv\Scripts\activate             # Windows
```

Always activate this before working. Your terminal prompt should show `(venv)`.

### 2.3 Create the Project Directory Structure

```bash

# Folder setup
mkdir -p douban-observatory/{notebooks,src/{client,db/{migrations},pipeline,training},models,data/raw}
cd douban-observatory

# Wheel setup (for dependencies)
python -m pip install --upgrade pip setuptools wheel
```

### 2.4 Install Dependencies

Create `requirements.txt`:

```
# Async HTTP and scraping
httpx==0.27.0
playwright==1.44.0
beautifulsoup4==4.12.3
lxml==5.2.2

# Database
asyncpg==0.29.0
sqlalchemy[asyncio]==2.0.30
psycopg2-binary==2.9.9
qdrant-client==1.9.1

# NLP — Chinese core
jieba==0.42.1
opencc-python-reimplemented==0.1.7

# NLP — Models and embeddings
transformers==4.41.2
torch==2.3.0
sentence-transformers==3.0.1
datasets==2.19.2
accelerate==0.30.0
peft==0.11.1

# Topic modeling
bertopic==0.16.2
umap-learn==0.5.6
hdbscan==0.8.38.post1

# Keyword extraction
scikit-learn==1.4.2
networkx==3.3

# Visualization
streamlit==1.35.0
plotly==5.21.0
wordcloud==1.9.3
pillow==10.3.0
matplotlib==3.8.4

# Scheduling
apscheduler==3.10.4

# Utilities
python-dotenv==1.0.1
tqdm==4.66.4
numpy==1.26.4
pandas==2.2.2
loguru==0.7.2
```

Install everything:

```bash
pip install -r requirements.txt
playwright install chromium
```

> **Note on torch:** The version above installs CPU torch. If you have a CUDA GPU locally, install the matching CUDA version from https://pytorch.org instead. Fine-tuning phases are designed to run on Kaggle's T4 GPUs, so local CPU is fine for all other phases.

### 2.5 Create the .env File

Create `.env` at the project root. This file holds all secrets and config — it must never be committed.

```
# Douban credentials
DOUBAN_USERNAME=your_douban_email
DOUBAN_PASSWORD=your_douban_password

# Database
POSTGRES_HOST=localhost
POSTGRES_PORT=5432
POSTGRES_DB=douban_observatory
POSTGRES_USER=observatory
POSTGRES_PASSWORD=choose_a_strong_password

# Qdrant
QDRANT_HOST=localhost
QDRANT_PORT=6333

# Target group (start with one)
TARGET_GROUP_ID=your_group_id_here

# Scrape interval in seconds
SCRAPE_INTERVAL=300
```

**How to find a Douban group ID:** Navigate to a group page in your browser. The URL looks like `https://www.douban.com/group/123456/`. The number is the group ID.

Create `.gitignore`:

```
.env
.env.*
venv/
__pycache__/
*.pyc
*.pyo
models/
data/raw/
*.log
.DS_Store
```

---

## 3. Phase 1 — Infrastructure: Docker, Postgres/TimescaleDB, Qdrant

**Goal:** Spin up the full production storage stack with a single command. By the end of this phase you will have a running Postgres instance with the TimescaleDB extension and a running Qdrant vector store, both containerized and persistent.

### 3.1 What Is TimescaleDB and Why Are We Using It?

TimescaleDB is a Postgres extension that adds first-class time-series support. It introduces **hypertables** — tables that automatically partition data by time behind the scenes. This means:

- Queries like "give me keyword frequency for the past 6 hours" are fast because Postgres only scans the relevant time partition, not the entire table
- You get continuous aggregates: pre-computed rollups (minute → hour → day) that update automatically as new data arrives
- It's 100% Postgres-compatible — you use the same SQL you already know

Without TimescaleDB, a 1-year table of NLP results queried by time window would do a full table scan every time your dashboard refreshes. That becomes unbearably slow around 1–10M rows.

### 3.2 What Is Qdrant and Why Are We Using It?

Qdrant is a purpose-built vector database. When your NLP pipeline generates sentence embeddings (dense floating-point vectors representing the meaning of a thread), you need to store them in a way that supports similarity search: "find all threads semantically similar to this one from the past week."

Storing 768-dimensional vectors as Postgres columns and querying them with cosine similarity is technically possible but slow and unindexed. Qdrant uses HNSW (a graph-based approximate nearest-neighbor algorithm) to make these queries fast at any scale.

### 3.3 Create `docker-compose.yml`

```yaml
version: '3.8'

services:
  timescaledb:
    image: timescale/timescaledb:latest-pg16
    container_name: observatory_db
    environment:
      POSTGRES_DB: ${POSTGRES_DB}
      POSTGRES_USER: ${POSTGRES_USER}
      POSTGRES_PASSWORD: ${POSTGRES_PASSWORD}
    ports:
      - "5432:5432"
    volumes:
      - timescaledb_data:/var/lib/postgresql/data
      - ./src/db/migrations:/docker-entrypoint-initdb.d  # Runs SQL on first start
    restart: unless-stopped
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U ${POSTGRES_USER} -d ${POSTGRES_DB}"]
      interval: 10s
      timeout: 5s
      retries: 5

  qdrant:
    image: qdrant/qdrant:v1.9.2
    container_name: observatory_qdrant
    ports:
      - "6333:6333"
      - "6334:6334"   # gRPC port — useful for high-throughput writes
    volumes:
      - qdrant_data:/qdrant/storage
    restart: unless-stopped

volumes:
  timescaledb_data:
  qdrant_data:
```

> **Why pin versions?** `latest` images can change unexpectedly and break your setup. Pinning Qdrant to `v1.9.2` and TimescaleDB to `latest-pg16` (which tracks the latest patch of PG16) gives you stability without manually managing minor patches.

### 3.4 Start the Stack

```bash
docker-compose up -d
```

The `-d` flag runs containers in the background. First run pulls images (~500MB total). Subsequent starts are instant.

Verify everything is running:

```bash
docker-compose ps
```

You should see both `observatory_db` and `observatory_qdrant` with status `Up`.

Check the database is ready:

```bash
docker exec -it observatory_db psql -U observatory -d douban_observatory -c "SELECT version();"
```

Verify TimescaleDB is installed:

```bash
docker exec -it observatory_db psql -U observatory -d douban_observatory -c "SELECT * FROM pg_extension WHERE extname = 'timescaledb';"
```

If the extension is not listed, enable it manually:

```bash
docker exec -it observatory_db psql -U observatory -d douban_observatory -c "CREATE EXTENSION IF NOT EXISTS timescaledb;"
```

Check Qdrant is running:

```bash
curl http://localhost:6333/healthz
# Should return: {"title":"qdrant - vector search engine","version":"1.9.2"}
```

### 3.5 Create `src/db/connection.py`

This module provides a shared async database engine for all other modules to import.

```python
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
    pool_size=10,           # Max 10 persistent connections
    max_overflow=20,        # Up to 20 extra connections under load
    pool_pre_ping=True,     # Test connections before using them
    echo=False              # Set True to log all SQL — useful for debugging
)

AsyncSessionLocal = sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False  # Keep objects accessible after commit
)

async def get_db():
    """Dependency-injection style session getter. Use as an async context manager."""
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
```

**Why async?** Your scraper, NLP pipeline, and scheduler all run on async event loops. Using synchronous database calls would block the entire event loop while waiting for a query to complete, defeating the purpose of async. `asyncpg` is a pure-async Postgres driver; `sqlalchemy[asyncio]` wraps it with the familiar SQLAlchemy interface.

### 3.6 Phase 1 Checklist

- [ ] Docker Desktop installed and running
- [ ] `docker-compose up -d` succeeds without errors
- [ ] `docker-compose ps` shows both containers as `Up`
- [ ] TimescaleDB extension confirmed present
- [ ] Qdrant health endpoint returns 200
- [ ] `src/db/connection.py` created
- [ ] `.env` created and populated
- [ ] `.gitignore` created

---

## 4. Phase 2 — The Douban Scraping Client

**Goal:** Build a clean, production-grade Python client that abstracts all Douban HTTP interaction. Your NLP pipeline and scheduler will call `client.get_group_threads(group_id)` — they will never touch raw HTTP.

### 4.1 Understanding Douban's Page Structure

Before writing a single line of scraping code, you need to understand what you're fetching. Douban group pages follow this URL pattern:

```
Group list:     https://www.douban.com/group/{group_id}/discussion
Thread page:    https://www.douban.com/group/topic/{topic_id}/
```

A group discussion page lists threads. Each thread entry contains:
- Thread title
- Author name and user ID
- Reply count
- Timestamp of last activity

A thread page contains:
- Original post (OP) body
- All replies, each with author, timestamp, and body text

**Important:** Some group content is visible without login. Content from private or members-only groups requires an authenticated session. We will build login support from the start.

### 4.2 Inspect the Real Page Structure

Before writing the parser, manually inspect Douban's current HTML in your browser:

1. Open `https://www.douban.com/group/{your_group_id}/discussion` in Chrome
2. Right-click → Inspect → find the thread list container
3. Note the CSS classes or HTML structure used for thread entries
4. Open one thread and note how replies are structured

> **Why inspect manually?** Douban has changed their HTML structure multiple times over the years. Any hardcoded class names in this tutorial may be outdated. Manual inspection against the live site takes 5 minutes and saves hours of debugging. The parsing logic you write in Step 4.5 must match what you actually see in DevTools, not what this tutorial assumes.

### 4.3 Create `src/client/session_manager.py`

```python
import httpx
import asyncio
import os
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# Headers that mimic a real browser. Without these, Douban may immediately
# return a bot-detection page instead of the actual content.
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Referer": "https://www.douban.com/",
}

class SessionManager:
    """
    Manages a persistent authenticated httpx session for Douban.
    Handles login, cookie persistence, and session refresh.
    """

    def __init__(self):
        self.client: httpx.AsyncClient | None = None
        self.authenticated = False

    async def initialize(self):
        """Create the httpx client and attempt login."""
        self.client = httpx.AsyncClient(
            headers=DEFAULT_HEADERS,
            follow_redirects=True,
            timeout=httpx.Timeout(30.0),    # 30s total timeout
            limits=httpx.Limits(
                max_keepalive_connections=5,
                max_connections=10
            )
        )
        await self._login()

    async def _login(self):
        """
        Log into Douban and store session cookies.

        Douban's login flow requires:
        1. GET the login page to retrieve the CSRF token (ck)
        2. POST credentials with the CSRF token

        This may need adjustment if Douban changes their login form.
        Inspect https://www.douban.com/login in DevTools → Network to see
        the actual form fields being submitted.
        """
        username = os.getenv("DOUBAN_USERNAME")
        password = os.getenv("DOUBAN_PASSWORD")

        if not username or not password:
            logger.warning("No Douban credentials found. Proceeding unauthenticated.")
            return

        try:
            # Step 1: Get the login page and extract CSRF token
            login_page = await self.client.get("https://www.douban.com/login")
            # The CSRF token is typically in a hidden input field named 'ck'
            # or in a cookie. Inspect the login page HTML to confirm the
            # exact mechanism — it changes periodically.

            # Step 2: Submit login form
            # Inspect the actual POST request in DevTools → Network → XHR
            # to see the exact field names. These are common but may vary:
            login_data = {
                "source": "None",
                "redir": "https://www.douban.com",
                "form_email": username,
                "form_password": password,
                "login": "登录",
            }

            response = await self.client.post(
                "https://www.douban.com/login",
                data=login_data,
            )

            # Verify login by checking for a user-specific cookie or redirect
            if "dbcl2" in self.client.cookies:
                self.authenticated = True
                logger.info("Douban login successful.")
            else:
                logger.warning(
                    "Login may have failed — 'dbcl2' cookie not found. "
                    "Inspect the login response HTML for an error message."
                )

        except Exception as e:
            logger.error(f"Login failed: {e}")

    async def get(self, url: str, **kwargs) -> httpx.Response:
        """
        Make an authenticated GET request with jittered rate limiting.

        The jitter (random delay variation) is important: predictable fixed
        intervals are easier for bot-detection systems to identify than
        randomized human-like timing.
        """
        import random

        # Random delay between 1.5 and 4.5 seconds between requests
        # Adjust these bounds if you get rate-limited (increase them)
        delay = random.uniform(1.5, 4.5)
        await asyncio.sleep(delay)

        try:
            response = await self.client.get(url, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as e:
            if e.response.status_code == 403:
                logger.warning(f"403 Forbidden on {url} — possible bot detection.")
            elif e.response.status_code == 429:
                logger.warning(f"429 Rate limited. Backing off for 60 seconds.")
                await asyncio.sleep(60)
            raise

    async def close(self):
        if self.client:
            await self.client.aclose()
```

### 4.4 Create `src/client/douban_client.py`

```python
import asyncio
import hashlib
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional
from bs4 import BeautifulSoup
from loguru import logger
from .session_manager import SessionManager


@dataclass
class Thread:
    """Represents a single Douban group thread."""
    thread_id: str
    group_id: str
    title: str
    author_id: str
    author_name: str
    body: str
    reply_count: int
    created_at: datetime
    last_active_at: datetime
    replies: list = field(default_factory=list)
    url: str = ""


@dataclass
class Reply:
    """Represents a single reply within a thread."""
    reply_id: str
    thread_id: str
    author_id: str
    author_name: str
    body: str
    created_at: datetime


class DoubanClient:
    """
    High-level Douban client. Callers interact with this class only —
    all HTTP and parsing details are hidden inside.

    Usage:
        client = DoubanClient()
        await client.initialize()
        threads = await client.get_group_threads("123456")
        for thread in threads:
            full_thread = await client.get_thread(thread.thread_id)
    """

    BASE_URL = "https://www.douban.com"

    def __init__(self):
        self.session = SessionManager()

    async def initialize(self):
        await self.session.initialize()

    async def get_group_threads(
        self,
        group_id: str,
        max_pages: int = 3
    ) -> list[Thread]:
        """
        Fetch the most recent threads from a group's discussion page.

        Args:
            group_id: The numeric Douban group ID
            max_pages: How many pages of thread listings to fetch.
                       Each page typically shows 25 threads.
                       3 pages = up to 75 threads per 5-minute cycle.

        Returns:
            List of Thread objects (without replies populated).
        """
        threads = []

        for page in range(max_pages):
            url = f"{self.BASE_URL}/group/{group_id}/discussion?start={page * 25}"
            logger.info(f"Fetching group {group_id} page {page + 1}")

            try:
                response = await self.session.get(url)
                page_threads = self._parse_thread_list(response.text, group_id)
                threads.extend(page_threads)

                # If a page returns fewer than 25 threads, we've hit the end
                if len(page_threads) < 25:
                    break

            except Exception as e:
                logger.error(f"Failed to fetch page {page + 1} for group {group_id}: {e}")
                break

        logger.info(f"Fetched {len(threads)} threads from group {group_id}")
        return threads

    async def get_thread(self, thread_id: str) -> Optional[Thread]:
        """
        Fetch a thread's full body and all replies.

        Args:
            thread_id: The numeric Douban topic ID

        Returns:
            Thread object with replies populated, or None if fetch fails.
        """
        url = f"{self.BASE_URL}/group/topic/{thread_id}/"
        logger.info(f"Fetching thread {thread_id}")

        try:
            response = await self.session.get(url)
            return self._parse_thread_page(response.text, thread_id)
        except Exception as e:
            logger.error(f"Failed to fetch thread {thread_id}: {e}")
            return None

    def _parse_thread_list(self, html: str, group_id: str) -> list[Thread]:
        """
        Parse the thread listing page HTML into Thread objects.

        !! IMPORTANT: Inspect the live Douban page in DevTools and confirm
        the CSS selectors below match what you see. Douban's HTML structure
        changes periodically. The selectors here are based on common patterns
        but may require adjustment.
        """
        soup = BeautifulSoup(html, "lxml")
        threads = []

        # The discussion table typically has class 'olt' or similar.
        # Inspect the live page and update this selector.
        table = soup.find("table", class_="olt")
        if not table:
            logger.warning(
                "Thread list table not found. "
                "Inspect the page HTML and update the selector in _parse_thread_list()."
            )
            return []

        rows = table.find_all("tr")[1:]  # Skip header row

        for row in rows:
            try:
                cells = row.find_all("td")
                if len(cells) < 4:
                    continue

                # Extract thread link and ID from the title cell
                title_cell = cells[0]
                link = title_cell.find("a")
                if not link:
                    continue

                href = link.get("href", "")
                # URL pattern: https://www.douban.com/group/topic/123456/
                thread_id = href.rstrip("/").split("/")[-1]
                title = link.get_text(strip=True)

                # Author cell
                author_cell = cells[1]
                author_link = author_cell.find("a")
                author_name = author_link.get_text(strip=True) if author_link else "unknown"
                author_id = ""
                if author_link:
                    # URL pattern: https://www.douban.com/people/username/
                    author_id = author_link.get("href", "").rstrip("/").split("/")[-1]

                # Reply count cell
                reply_cell = cells[2]
                try:
                    reply_count = int(reply_cell.get_text(strip=True))
                except ValueError:
                    reply_count = 0

                # Last activity timestamp cell
                time_cell = cells[3]
                time_str = time_cell.get_text(strip=True)
                last_active_at = self._parse_douban_time(time_str)

                thread = Thread(
                    thread_id=thread_id,
                    group_id=group_id,
                    title=title,
                    author_id=author_id,
                    author_name=author_name,
                    body="",  # Body populated in get_thread()
                    reply_count=reply_count,
                    created_at=last_active_at,  # Approximate; updated in get_thread()
                    last_active_at=last_active_at,
                    url=href
                )
                threads.append(thread)

            except Exception as e:
                logger.debug(f"Failed to parse row: {e}")
                continue

        return threads

    def _parse_thread_page(self, html: str, thread_id: str) -> Optional[Thread]:
        """
        Parse a full thread page into a Thread object with replies.

        !! IMPORTANT: Same caveat as _parse_thread_list — inspect the live
        page and confirm the selectors match what you see.
        """
        soup = BeautifulSoup(html, "lxml")

        # Thread body is typically in a div with class 'topic-content' or similar
        # Inspect and update:
        body_div = soup.find("div", class_="topic-content")
        if not body_div:
            # Try fallback selectors
            body_div = soup.find("div", attrs={"id": "link-report"})
        if not body_div:
            logger.warning(f"Could not find thread body for {thread_id}")
            return None

        body = body_div.get_text(separator="\n", strip=True)

        # Parse thread metadata from the topic header
        # Author, title, creation time — inspect and update selectors
        header = soup.find("div", class_="topic-doc")
        title = ""
        author_name = ""
        author_id = ""
        created_at = datetime.utcnow()

        if header:
            h1 = header.find("h1")
            if h1:
                title = h1.get_text(strip=True)
            author_link = header.find("a", class_="author")
            if author_link:
                author_name = author_link.get_text(strip=True)
                author_id = author_link.get("href", "").rstrip("/").split("/")[-1]
            time_span = header.find("span", class_="pubtime")
            if time_span:
                created_at = self._parse_douban_time(time_span.get_text(strip=True))

        # Parse replies
        replies = []
        # Replies are typically in a list with class 'topic-reply' or 'reply-list'
        # Inspect and update:
        reply_items = soup.find_all("li", class_="reply-item")
        for item in reply_items:
            try:
                reply_id = item.get("data-cid", "")
                reply_author_link = item.find("a", class_="reply-author")
                reply_author_name = reply_author_link.get_text(strip=True) if reply_author_link else ""
                reply_author_id = ""
                if reply_author_link:
                    reply_author_id = reply_author_link.get("href", "").rstrip("/").split("/")[-1]

                reply_content = item.find("p", class_="reply-content")
                reply_body = reply_content.get_text(separator="\n", strip=True) if reply_content else ""

                reply_time_span = item.find("span", class_="reply-time")
                reply_time = self._parse_douban_time(
                    reply_time_span.get_text(strip=True) if reply_time_span else ""
                )

                if reply_id and reply_body:
                    replies.append(Reply(
                        reply_id=reply_id,
                        thread_id=thread_id,
                        author_id=reply_author_id,
                        author_name=reply_author_name,
                        body=reply_body,
                        created_at=reply_time
                    ))
            except Exception as e:
                logger.debug(f"Failed to parse reply: {e}")
                continue

        return Thread(
            thread_id=thread_id,
            group_id="",  # Caller knows the group_id
            title=title,
            author_id=author_id,
            author_name=author_name,
            body=body,
            reply_count=len(replies),
            created_at=created_at,
            last_active_at=created_at,
            replies=replies,
            url=f"https://www.douban.com/group/topic/{thread_id}/"
        )

    def _parse_douban_time(self, time_str: str) -> datetime:
        """
        Parse Douban's various timestamp formats into a datetime object.

        Douban uses several formats:
        - "2024-05-01 14:30:00"  (full timestamp)
        - "05-01 14:30"          (current year, short)
        - "14:30"                (today)
        - "3天前"                 (relative: 3 days ago)
        """
        from datetime import timedelta
        import re

        time_str = time_str.strip()
        now = datetime.utcnow()

        # Full timestamp
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M"):
            try:
                return datetime.strptime(time_str, fmt)
            except ValueError:
                continue

        # Short date (current year implied)
        try:
            return datetime.strptime(f"{now.year}-{time_str}", "%Y-%m-%d %H:%M")
        except ValueError:
            pass

        # Today, time only
        try:
            t = datetime.strptime(time_str, "%H:%M")
            return now.replace(hour=t.hour, minute=t.minute, second=0, microsecond=0)
        except ValueError:
            pass

        # Relative: "X天前"
        match = re.match(r"(\d+)天前", time_str)
        if match:
            return now - timedelta(days=int(match.group(1)))

        # Relative: "X小时前"
        match = re.match(r"(\d+)小时前", time_str)
        if match:
            return now - timedelta(hours=int(match.group(1)))

        # Fallback
        logger.debug(f"Could not parse timestamp: '{time_str}', using now")
        return now

    async def close(self):
        await self.session.close()
```

### 4.5 Validate in `notebooks/01_scraper_exploration.ipynb`

Open the notebook and run these cells to verify the client before proceeding.

#### Cell 1 — Basic Fetch Test

```python
import asyncio
import sys
sys.path.append('../src')

from client.douban_client import DoubanClient

async def test_fetch():
    client = DoubanClient()
    await client.initialize()

    # Replace with your target group ID
    GROUP_ID = "your_group_id_here"

    threads = await client.get_group_threads(GROUP_ID, max_pages=1)
    print(f"Fetched {len(threads)} threads")

    for t in threads[:5]:
        print(f"  [{t.thread_id}] {t.title} — {t.author_name} ({t.reply_count} replies)")

    await client.close()

await test_fetch()
```

**What to look for:** You should see 5 thread titles printed. If you see 0 threads, the parser is not finding the table. Print the raw HTML and inspect the actual structure:

```python
async def debug_html():
    client = DoubanClient()
    await client.initialize()
    response = await client.session.get(
        f"https://www.douban.com/group/your_group_id/discussion"
    )
    print(response.text[:5000])  # Print first 5000 chars
    await client.close()

await debug_html()
```

Compare the printed HTML against the selectors in `_parse_thread_list()` and update as needed.

#### Cell 2 — Full Thread Fetch Test

```python
async def test_thread_fetch():
    client = DoubanClient()
    await client.initialize()

    GROUP_ID = "your_group_id_here"
    threads = await client.get_group_threads(GROUP_ID, max_pages=1)

    if threads:
        # Fetch the first thread fully
        full_thread = await client.get_thread(threads[0].thread_id)
        if full_thread:
            print(f"Title: {full_thread.title}")
            print(f"Body preview: {full_thread.body[:300]}")
            print(f"Replies: {len(full_thread.replies)}")
            if full_thread.replies:
                print(f"First reply: {full_thread.replies[0].body[:200]}")

    await client.close()

await test_thread_fetch()
```

**What to look for:** A populated title, body text (Chinese characters), and at least some replies if the thread has them.

### 4.6 Phase 2 Checklist

- [ ] `session_manager.py` created and login tested
- [ ] `douban_client.py` created
- [ ] HTML selectors in `_parse_thread_list()` validated against the live page
- [ ] HTML selectors in `_parse_thread_page()` validated against the live page
- [ ] Notebook Cell 1 returns at least one thread
- [ ] Notebook Cell 2 returns a thread with Chinese body text
- [ ] Timestamps are parsed correctly (not all defaulting to now)

---

## 5. Phase 3 — Storage Schema Design

**Goal:** Design and create the database schema that will hold a year of scraped threads, NLP results, and time-aggregated metrics. Get this right now — changing schema after data is ingested is painful.

### 5.1 Schema Design Principles

The schema follows one rule: **the Streamlit dashboard never touches raw tables at query time.** Everything the dashboard displays is pre-computed and written to aggregate tables by the NLP pipeline. This separation is what makes the dashboard fast regardless of how much data accumulates.

The data flows in one direction:

```
groups → threads → replies (raw data, append-only)
                ↓
        nlp_results (per-thread NLP output)
                ↓
        keyword_timeseries   (hourly rollups)
        sentiment_timeseries (hourly rollups)
        topic_timeseries     (hourly rollups)
```

### 5.2 Create `src/db/migrations/001_initial.sql`

This SQL file runs automatically when Postgres starts for the first time (via the Docker volume mount). Read every table definition and its comments carefully — they encode important architectural decisions.

```sql
-- Enable TimescaleDB extension
CREATE EXTENSION IF NOT EXISTS timescaledb;

-- ============================================================
-- GROUPS
-- ============================================================
CREATE TABLE IF NOT EXISTS groups (
    group_id        TEXT PRIMARY KEY,
    name            TEXT,
    description     TEXT,
    member_count    INTEGER,
    created_at      TIMESTAMPTZ DEFAULT NOW(),
    last_scraped_at TIMESTAMPTZ
);

-- ============================================================
-- THREADS
-- Raw thread data. Append-only. Never update body after insert.
-- If a thread is edited on Douban, we store the original.
-- ============================================================
CREATE TABLE IF NOT EXISTS threads (
    thread_id       TEXT PRIMARY KEY,
    group_id        TEXT REFERENCES groups(group_id),
    title           TEXT NOT NULL,
    author_id       TEXT,
    author_name     TEXT,
    body            TEXT,
    reply_count     INTEGER DEFAULT 0,
    url             TEXT,
    created_at      TIMESTAMPTZ NOT NULL,  -- Original post time
    last_active_at  TIMESTAMPTZ NOT NULL,  -- Time of most recent reply
    scraped_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_threads_group_id ON threads(group_id);
CREATE INDEX idx_threads_created_at ON threads(created_at DESC);

-- ============================================================
-- REPLIES
-- Individual replies within threads.
-- ============================================================
CREATE TABLE IF NOT EXISTS replies (
    reply_id        TEXT PRIMARY KEY,
    thread_id       TEXT REFERENCES threads(thread_id) ON DELETE CASCADE,
    author_id       TEXT,
    author_name     TEXT,
    body            TEXT NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL,
    scraped_at      TIMESTAMPTZ DEFAULT NOW()
);

CREATE INDEX idx_replies_thread_id ON replies(thread_id);
CREATE INDEX idx_replies_created_at ON replies(created_at DESC);

-- ============================================================
-- NLP RESULTS
-- One row per thread per pipeline run.
-- sentiment_label: 'positive', 'negative', 'neutral'
-- sentiment_score: confidence [0, 1]
-- topic_id: assigned by BERTopic (-1 = noise/outlier)
-- topic_label: human-readable topic name
-- keywords: JSON array of {word, score} objects
-- summary: generated summary from LoRA model (nullable)
-- embedding_id: UUID reference into Qdrant collection
-- pipeline_version: allows reprocessing with new models
-- ============================================================
CREATE TABLE IF NOT EXISTS nlp_results (
    id                  BIGSERIAL PRIMARY KEY,
    thread_id           TEXT REFERENCES threads(thread_id) ON DELETE CASCADE,
    sentiment_label     TEXT,
    sentiment_score     FLOAT,
    topic_id            INTEGER,
    topic_label         TEXT,
    keywords            JSONB,   -- [{"word": "...", "score": 0.82}, ...]
    summary             TEXT,
    embedding_id        TEXT,    -- Qdrant point ID
    pipeline_version    TEXT,
    processed_at        TIMESTAMPTZ DEFAULT NOW()
);

-- Convert to hypertable partitioned by processed_at
-- TimescaleDB will automatically chunk this into weekly partitions
SELECT create_hypertable('nlp_results', 'processed_at',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);

CREATE INDEX idx_nlp_thread_id ON nlp_results(thread_id, processed_at DESC);

-- ============================================================
-- KEYWORD TIMESERIES
-- Pre-aggregated keyword frequency per time bucket.
-- This is what the lexical cloud and keyword trend charts query.
-- bucket: rounded to 1-hour intervals
-- word: segmented Chinese word
-- frequency: how many threads in this hour used this word
-- avg_tfidf: average TF-IDF score across those threads
-- ============================================================
CREATE TABLE IF NOT EXISTS keyword_timeseries (
    bucket          TIMESTAMPTZ NOT NULL,
    group_id        TEXT NOT NULL,
    word            TEXT NOT NULL,
    frequency       INTEGER,
    avg_tfidf       FLOAT,
    PRIMARY KEY (bucket, group_id, word)
);

SELECT create_hypertable('keyword_timeseries', 'bucket',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);

-- ============================================================
-- SENTIMENT TIMESERIES
-- Pre-aggregated sentiment per time bucket.
-- avg_score: mean sentiment score in the hour
-- positive_count, negative_count, neutral_count: raw counts
-- ============================================================
CREATE TABLE IF NOT EXISTS sentiment_timeseries (
    bucket          TIMESTAMPTZ NOT NULL,
    group_id        TEXT NOT NULL,
    avg_score       FLOAT,
    positive_count  INTEGER DEFAULT 0,
    negative_count  INTEGER DEFAULT 0,
    neutral_count   INTEGER DEFAULT 0,
    total_count     INTEGER DEFAULT 0,
    PRIMARY KEY (bucket, group_id)
);

SELECT create_hypertable('sentiment_timeseries', 'bucket',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);

-- ============================================================
-- TOPIC TIMESERIES
-- Pre-aggregated topic distribution per time bucket.
-- ============================================================
CREATE TABLE IF NOT EXISTS topic_timeseries (
    bucket          TIMESTAMPTZ NOT NULL,
    group_id        TEXT NOT NULL,
    topic_id        INTEGER NOT NULL,
    topic_label     TEXT,
    thread_count    INTEGER DEFAULT 0,
    PRIMARY KEY (bucket, group_id, topic_id)
);

SELECT create_hypertable('topic_timeseries', 'bucket',
    chunk_time_interval => INTERVAL '1 week',
    if_not_exists => TRUE
);

-- ============================================================
-- SCRAPE LOG
-- Records every scrape run for monitoring and debugging.
-- ============================================================
CREATE TABLE IF NOT EXISTS scrape_log (
    id              BIGSERIAL PRIMARY KEY,
    group_id        TEXT,
    started_at      TIMESTAMPTZ DEFAULT NOW(),
    finished_at     TIMESTAMPTZ,
    threads_found   INTEGER DEFAULT 0,
    threads_new     INTEGER DEFAULT 0,
    replies_new     INTEGER DEFAULT 0,
    error           TEXT        -- NULL if success
);
```

### 5.3 Apply and Validate the Schema

The migration file runs automatically on first Docker start because of the volume mount in `docker-compose.yml`. If you need to re-apply it manually:

```bash
docker exec -i observatory_db psql -U observatory -d douban_observatory \
  < src/db/migrations/001_initial.sql
```

Verify all tables were created:

```bash
docker exec -it observatory_db psql -U observatory -d douban_observatory \
  -c "\dt"
```

Verify hypertables were created:

```bash
docker exec -it observatory_db psql -U observatory -d douban_observatory \
  -c "SELECT hypertable_name, num_chunks FROM timescaledb_information.hypertables;"
```

You should see `nlp_results`, `keyword_timeseries`, `sentiment_timeseries`, and `topic_timeseries` listed.

### 5.4 Create `src/db/models.py`

SQLAlchemy ORM models that map Python classes to the schema above. These are used by the scraper and pipeline to insert data without writing raw SQL.

```python
from datetime import datetime
from sqlalchemy import Column, String, Integer, Float, Text, DateTime, BigInteger, JSON
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMPTZ
from sqlalchemy.orm import DeclarativeBase

class Base(DeclarativeBase):
    pass

class Group(Base):
    __tablename__ = "groups"
    group_id = Column(String, primary_key=True)
    name = Column(String)
    description = Column(Text)
    member_count = Column(Integer)
    created_at = Column(TIMESTAMPTZ, default=datetime.utcnow)
    last_scraped_at = Column(TIMESTAMPTZ)

class Thread(Base):
    __tablename__ = "threads"
    thread_id = Column(String, primary_key=True)
    group_id = Column(String)
    title = Column(String, nullable=False)
    author_id = Column(String)
    author_name = Column(String)
    body = Column(Text)
    reply_count = Column(Integer, default=0)
    url = Column(String)
    created_at = Column(TIMESTAMPTZ, nullable=False)
    last_active_at = Column(TIMESTAMPTZ, nullable=False)
    scraped_at = Column(TIMESTAMPTZ, default=datetime.utcnow)

class Reply(Base):
    __tablename__ = "replies"
    reply_id = Column(String, primary_key=True)
    thread_id = Column(String)
    author_id = Column(String)
    author_name = Column(String)
    body = Column(Text, nullable=False)
    created_at = Column(TIMESTAMPTZ, nullable=False)
    scraped_at = Column(TIMESTAMPTZ, default=datetime.utcnow)

class NLPResult(Base):
    __tablename__ = "nlp_results"
    id = Column(BigInteger, primary_key=True, autoincrement=True)
    thread_id = Column(String)
    sentiment_label = Column(String)
    sentiment_score = Column(Float)
    topic_id = Column(Integer)
    topic_label = Column(String)
    keywords = Column(JSONB)
    summary = Column(Text)
    embedding_id = Column(String)
    pipeline_version = Column(String)
    processed_at = Column(TIMESTAMPTZ, default=datetime.utcnow)
```

### 5.5 Validate in `notebooks/02_schema_validation.ipynb`

```python
import asyncio
import sys
sys.path.append('../src')

from db.connection import engine
from db.models import Base, Group, Thread
from sqlalchemy import text

async def validate_schema():
    # Test connection
    async with engine.begin() as conn:
        result = await conn.execute(text("SELECT version()"))
        print("Postgres version:", result.scalar())

        # Check hypertables
        result = await conn.execute(text(
            "SELECT hypertable_name FROM timescaledb_information.hypertables"
        ))
        hypertables = [row[0] for row in result.fetchall()]
        print("Hypertables:", hypertables)

        # Test a simple insert + select cycle
        await conn.execute(text(
            "INSERT INTO groups (group_id, name) VALUES ('test_001', 'Test Group') "
            "ON CONFLICT DO NOTHING"
        ))
        result = await conn.execute(text("SELECT * FROM groups WHERE group_id = 'test_001'"))
        row = result.fetchone()
        print("Test row:", row)

await validate_schema()
```

**What to look for:** Postgres version printed, 4 hypertable names listed, test row returned correctly.

### 5.6 Phase 3 Checklist

- [ ] `001_initial.sql` created and applied
- [ ] All tables visible in `\dt`
- [ ] 4 hypertables confirmed in TimescaleDB info view
- [ ] `models.py` created
- [ ] Notebook validation passes: connection, hypertables, insert/select cycle

---

## 6. Phase 4 — Chinese Text Preprocessing Pipeline

**Goal:** Build the text cleaning and segmentation module that every downstream NLP step depends on. This phase explains why Chinese NLP preprocessing is fundamentally different from English and establishes the foundation that makes your models work.

### 6.1 Why Chinese NLP Preprocessing Is Different

In English, words are separated by spaces: `"the cat sat on the mat"` → 6 tokens, no work required.

In Chinese, there are no spaces: `"猫坐在垫子上"` is 7 characters that form 5 words. Before any NLP can happen, you must first **segment** the text into words — a task that does not exist in English pipelines.

Additionally, Chinese NLP must handle:

| Challenge | Example | Solution |
|---|---|---|
| No word boundaries | `猫坐在垫子上` | Word segmentation (jieba) |
| Traditional vs Simplified | `貓` vs `猫` | OpenCC conversion |
| Mixed scripts | `今天lol真的太难了` | Script-aware cleaning |
| Internet slang | `666`, `yyds`, `破防` | Domain-specific handling |
| Emoji and punctuation | `😭😭😭!!!` | Selective stripping |
| Doubled characters | `好好好好好` | Normalization |
| Full-width characters | `！？　` | Half-width conversion |

Douban social text contains all of these. Skipping any of them degrades every downstream model.

### 6.2 Download the Chinese Stopword List

```bash
# Download a comprehensive Chinese stopword list
curl -L "https://raw.githubusercontent.com/goto456/stopwords/master/cn_stopwords.txt" \
  -o data/stopwords_zh.txt
```

Inspect it:
```bash
head -20 data/stopwords_zh.txt
wc -l data/stopwords_zh.txt
```

You should see ~2000 common Chinese function words and punctuation that carry no semantic information: `的`, `了`, `是`, `在`, `我`, `他`, etc.

> **Note on social media stopwords:** The downloaded list is tuned for formal text. Social media adds additional meaningless tokens: `哈哈哈`, `嗯嗯`, `好的好的`. You will extend the stopword list with domain-specific entries as you accumulate data.

### 6.3 Create `src/pipeline/preprocessor.py`

```python
import re
import jieba
import jieba.analyse
import opencc
from pathlib import Path
from loguru import logger

# Initialize OpenCC converter: Traditional → Simplified Chinese
# 't2s' = Traditional to Simplified (covers Hong Kong/Taiwan text)
_converter = opencc.OpenCC('t2s')

# Load stopwords
_STOPWORDS_PATH = Path(__file__).parent.parent.parent / "data" / "stopwords_zh.txt"
_STOPWORDS: set[str] = set()

if _STOPWORDS_PATH.exists():
    with open(_STOPWORDS_PATH, encoding="utf-8") as f:
        _STOPWORDS = {line.strip() for line in f if line.strip()}
    logger.info(f"Loaded {len(_STOPWORDS)} stopwords")
else:
    logger.warning(f"Stopwords file not found at {_STOPWORDS_PATH}")

# Tell jieba to use stopwords during extraction
jieba.analyse.set_stop_words(str(_STOPWORDS_PATH))

# Precompile regex patterns — compiling once at module load is faster than
# recompiling on every function call
_RE_URL = re.compile(
    r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+"
)
_RE_HTML_TAG = re.compile(r"<[^>]+>")
_RE_AT_MENTION = re.compile(r"@[\w\u4e00-\u9fff]+")  # @用户名
_RE_TOPIC_TAG = re.compile(r"#[^#]+#")                # #话题标签#
_RE_EMOJI_BRACKET = re.compile(r"\[[\w\u4e00-\u9fff]+\]")  # Douban emoji [微笑]
_RE_REPEATED_CHARS = re.compile(r"(.)\1{3,}")          # 4+ consecutive same chars
_RE_FULL_WIDTH = str.maketrans(
    "！？，。；：""''（）【】《》",
    "!?,. ;:\"\"''()[]<>"
)
_RE_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """
    Step 1: Normalize raw Douban text.

    This function converts the raw scraped text into a clean,
    consistent form before segmentation. It does NOT remove
    semantic content — that happens in the segmentation step.

    Args:
        text: Raw text from Douban thread body or reply

    Returns:
        Normalized text string
    """
    if not text or not text.strip():
        return ""

    # Convert Traditional → Simplified Chinese
    # This is important because some users write in Traditional
    text = _converter.convert(text)

    # Strip HTML tags (sometimes present in body text)
    text = _RE_HTML_TAG.sub("", text)

    # Remove URLs
    text = _RE_URL.sub("", text)

    # Remove @mentions (not semantically meaningful for topic analysis)
    text = _RE_AT_MENTION.sub("", text)

    # Remove #topic tags# — the topic label itself is noise for body analysis
    text = _RE_TOPIC_TAG.sub("", text)

    # Remove Douban bracket emojis like [微笑] [怒]
    text = _RE_EMOJI_BRACKET.sub("", text)

    # Remove non-Chinese, non-ASCII characters (emoji, special symbols)
    # Keep: Chinese chars \u4e00-\u9fff, ASCII letters/numbers, common punctuation
    text = re.sub(r"[^\u4e00-\u9fff\u3400-\u4dbf\ufa0e\ufa0f\ufa11\ufa13\ufa14"
                  r"\ufa1f\ufa21\ufa23\ufa24\ufa27-\ufa29"  # CJK extensions
                  r"a-zA-Z0-9\s\.,!?;:\-\(\)\[\]\"\'，。！？；：、\n]", "", text)

    # Normalize full-width punctuation to ASCII
    text = text.translate(_RE_FULL_WIDTH)

    # Collapse 4+ repeated characters to 3 (e.g., "哈哈哈哈哈哈" → "哈哈哈")
    # This reduces noise while preserving some expressiveness
    text = _RE_REPEATED_CHARS.sub(r"\1\1\1", text)

    # Collapse whitespace
    text = _RE_WHITESPACE.sub(" ", text).strip()

    return text


def segment(text: str,
            remove_stopwords: bool = True,
            min_word_len: int = 2,
            keep_english: bool = True) -> list[str]:
    """
    Step 2: Segment normalized Chinese text into a list of words.

    Uses jieba's accurate mode (HMM-based segmentation).
    Filters stopwords, short single characters, and optionally English.

    Args:
        text: Normalized text from normalize_text()
        remove_stopwords: Whether to filter the stopword list
        min_word_len: Minimum character length for Chinese words (1 = keep singles)
        keep_english: Whether to keep English words (set False for pure Chinese analysis)

    Returns:
        List of segmented, filtered words
    """
    if not text:
        return []

    words = jieba.lcut(text, cut_all=False)  # Accurate mode, not full-mode

    result = []
    for word in words:
        word = word.strip()
        if not word:
            continue

        # Keep English words if flag is set
        if re.match(r"^[a-zA-Z][a-zA-Z0-9]*$", word):
            if keep_english and len(word) >= 2:
                result.append(word.lower())
            continue

        # Skip pure numbers and punctuation
        if re.match(r"^[\d\s\.,!?;:\-\(\)\[\]\"\'，。！？；：、]+$", word):
            continue

        # Skip stopwords
        if remove_stopwords and word in _STOPWORDS:
            continue

        # Skip single characters (almost always noise in social text)
        if len(word) < min_word_len:
            continue

        result.append(word)

    return result


def preprocess(text: str, **kwargs) -> dict:
    """
    Full preprocessing pipeline: normalize then segment.

    Returns both the normalized text and the token list so callers
    can use whichever representation they need.

    Args:
        text: Raw Douban text
        **kwargs: Passed through to segment()

    Returns:
        {
            "raw": original text,
            "normalized": cleaned text,
            "tokens": list of segmented words
        }
    """
    normalized = normalize_text(text)
    tokens = segment(normalized, **kwargs)
    return {
        "raw": text,
        "normalized": normalized,
        "tokens": tokens
    }


def add_domain_stopwords(words: list[str]) -> None:
    """
    Add domain-specific stopwords discovered from your Douban data.
    Call this with frequent but meaningless words you observe.

    Example:
        add_domain_stopwords(["楼主", "回复", "支持", "哈哈哈"])
    """
    _STOPWORDS.update(words)
    logger.info(f"Added {len(words)} domain stopwords. Total: {len(_STOPWORDS)}")
```

### 6.4 Validate in `notebooks/03_preprocessing_exploration.ipynb`

#### Cell 1 — Basic Normalization Test

```python
import sys
sys.path.append('../src')
from pipeline.preprocessor import normalize_text, segment, preprocess

# Test with a Douban-style post
raw_text = """
楼主说的有道理！[微笑] 我昨天去看了，真的太好看了哈哈哈哈哈哈
@豆友小明 你也去看了吗？ #电影推荐#
https://movie.douban.com/subject/123456/
666666 这部片子yyds了
"""

print("=== Normalized ===")
normalized = normalize_text(raw_text)
print(repr(normalized))
```

**What to look for:** URLs removed, emoji brackets removed, @mentions removed, #topics# removed, repeated 哈 collapsed to 哈哈哈, full-width punctuation converted.

#### Cell 2 — Segmentation Test

```python
print("=== Tokens ===")
tokens = segment(normalized)
print(tokens)
print(f"\nToken count: {len(tokens)}")
```

**What to look for:** Chinese words correctly split (not individual characters), stopwords removed, short single characters absent.

#### Cell 3 — Real Data Test

```python
# Use actual scraped text from your notebook 01 if available
# or paste a real Douban reply here
real_reply = "真的，这个电影的剧情逻辑太混乱了，导演完全不知道自己在讲什么故事，白白浪费了这么好的演员阵容"

result = preprocess(real_reply)
print("Tokens:", result["tokens"])
```

**What to look for:** `电影`, `剧情`, `逻辑`, `导演`, `演员` should appear as tokens. `真的`, `这个`, `了` should be filtered as stopwords.

#### Cell 4 — Identify Domain Stopwords

```python
from collections import Counter

# Run on a batch of real scraped text (if available from Phase 2)
# and find high-frequency low-information words
sample_texts = [
    "楼主你好啊 支持一下",
    "回复一下表示存在",
    "顶一下 强烈支持楼主",
    "楼上说得对 深有同感",
]

all_tokens = []
for t in sample_texts:
    result = preprocess(t)
    all_tokens.extend(result["tokens"])

freq = Counter(all_tokens).most_common(30)
print("Most frequent tokens in sample:")
for word, count in freq:
    print(f"  {word}: {count}")
```

**What to look for:** Words like `楼主`, `回复`, `支持`, `顶` appear frequently but carry no semantic information. Add them to your domain stopwords:

```python
from pipeline.preprocessor import add_domain_stopwords
add_domain_stopwords(["楼主", "回复", "支持", "顶", "感谢", "谢谢", "哈哈", "同感"])
```

### 6.5 Phase 4 Checklist

- [ ] OpenCC library installed and Traditional→Simplified conversion working
- [ ] Stopwords file downloaded (2000+ entries)
- [ ] `preprocessor.py` created
- [ ] Normalization removes URLs, emojis, @mentions, #topics
- [ ] Repeated characters collapsed correctly
- [ ] Segmentation produces multi-character tokens (not single chars)
- [ ] Stopwords filtered correctly
- [ ] Domain stopwords identified from real data and added

---

## 7. Phase 5 — Keyword Extraction & Lexical Analysis

**Goal:** Extract meaningful keywords from threads using TF-IDF and TextRank, aggregate them by time window, and build the lexical cloud visualization layer.

### 7.1 Two Approaches to Keyword Extraction

**TF-IDF (Term Frequency-Inverse Document Frequency)**
Scores a word based on how often it appears in a specific document relative to how rare it is across the entire corpus. A word that appears a lot in one thread but rarely in other threads gets a high score — it's distinctive.

The formula: `TF-IDF(w, d) = (count of w in d / total words in d) × log(total docs / docs containing w)`

TF-IDF is fast, interpretable, and works well for finding topically distinctive words. Its weakness: it treats words as independent and doesn't understand that `电影` and `影片` mean the same thing.

**TextRank**
Treats words as nodes in a graph. Two words are connected if they co-occur within a sliding window. Words that are highly connected to many other important words get high scores. This is analogous to PageRank applied to text.

TextRank finds words that are central to the semantic structure of a document, not just words that are frequent. It tends to surface more thematically coherent keywords than TF-IDF.

**We use both.** TF-IDF for the time-series aggregates (fast, good for trending word detection). TextRank for per-thread keyword extraction (better quality, used for the lexical cloud).

### 7.2 Create `src/pipeline/keywords.py`

```python
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba.analyse
from loguru import logger
from .preprocessor import preprocess


def extract_keywords_textrank(text: str, topk: int = 10) -> list[dict]:
    """
    Extract keywords using jieba's TextRank implementation.

    Args:
        text: Raw or normalized text (we normalize internally)
        topk: Number of keywords to return

    Returns:
        List of {"word": str, "score": float} dicts, sorted by score descending
    """
    if not text or not text.strip():
        return []

    # jieba.analyse.textrank handles its own segmentation internally
    # We pass the normalized text to remove noise first
    from .preprocessor import normalize_text
    normalized = normalize_text(text)

    keywords = jieba.analyse.textrank(
        normalized,
        topK=topk,
        withWeight=True,    # Return (word, score) tuples
        allowPOS=(
            "ns",  # Place names
            "n",   # Common nouns
            "vn",  # Verbal nouns
            "v",   # Verbs (important in Chinese social text)
            "nr",  # Person names
            "eng", # English words
        )
    )

    return [{"word": word, "score": round(score, 4)} for word, score in keywords]


def extract_keywords_tfidf(text: str, topk: int = 10) -> list[dict]:
    """
    Extract keywords using jieba's TF-IDF implementation.

    Note: jieba's TF-IDF uses a pre-built IDF corpus from news/web text.
    For domain-specific accuracy, we will retrain this IDF in Phase 13
    using your own Douban corpus. For now, the default is sufficient.
    """
    if not text or not text.strip():
        return []

    from .preprocessor import normalize_text
    normalized = normalize_text(text)

    keywords = jieba.analyse.extract_tags(
        normalized,
        topK=topk,
        withWeight=True,
        allowPOS=("ns", "n", "vn", "v", "nr", "eng")
    )

    return [{"word": word, "score": round(score, 4)} for word, score in keywords]


def extract_keywords_combined(text: str, topk: int = 10) -> list[dict]:
    """
    Combine TextRank and TF-IDF scores for better keyword quality.

    Strategy:
    - Run both extractors with 2x topk candidates
    - Normalize each score list to [0, 1]
    - Average the two scores
    - Return top-k by combined score

    Words that rank highly in both methods are the most reliable keywords.
    """
    n = topk * 2

    tr_results = extract_keywords_textrank(text, topk=n)
    tf_results = extract_keywords_tfidf(text, topk=n)

    # Build score dicts
    tr_scores = {r["word"]: r["score"] for r in tr_results}
    tf_scores = {r["word"]: r["score"] for r in tf_results}

    # Normalize each set to [0, 1]
    def normalize_scores(scores_dict):
        if not scores_dict:
            return {}
        max_score = max(scores_dict.values()) or 1.0
        return {w: s / max_score for w, s in scores_dict.items()}

    tr_norm = normalize_scores(tr_scores)
    tf_norm = normalize_scores(tf_scores)

    # Combine: all words from either method
    all_words = set(tr_norm.keys()) | set(tf_norm.keys())
    combined = {}
    for word in all_words:
        tr_s = tr_norm.get(word, 0)
        tf_s = tf_norm.get(word, 0)
        # Words found by both methods get a bonus
        bonus = 0.1 if (word in tr_norm and word in tf_norm) else 0
        combined[word] = (tr_s + tf_s) / 2 + bonus

    # Sort and return topk
    sorted_words = sorted(combined.items(), key=lambda x: x[1], reverse=True)
    return [{"word": w, "score": round(s, 4)} for w, s in sorted_words[:topk]]


class CorpusTFIDF:
    """
    Corpus-level TF-IDF for time-window keyword aggregation.

    Instead of running per-thread TF-IDF (which has no IDF component for
    small single documents), this class builds a vectorizer across all
    threads in a time window, giving meaningful IDF scores.

    Usage:
        tfidf = CorpusTFIDF()
        tfidf.fit(list_of_thread_texts)
        top_words = tfidf.top_words_by_time_window(threads_df, window='1H')
    """

    def __init__(self, max_features: int = 5000):
        self.vectorizer = TfidfVectorizer(
            tokenizer=self._tokenize,
            token_pattern=None,     # We handle tokenization manually
            max_features=max_features,
            min_df=2,               # Word must appear in at least 2 threads
            max_df=0.85,            # Ignore words in >85% of threads (too common)
            sublinear_tf=True,      # Use log(TF) to reduce impact of very high frequency
        )
        self.fitted = False

    def _tokenize(self, text: str) -> list[str]:
        """Tokenizer for sklearn — returns list of tokens."""
        result = preprocess(text)
        return result["tokens"]

    def fit(self, texts: list[str]) -> None:
        """Fit the vectorizer on a corpus of texts."""
        if len(texts) < 5:
            logger.warning("Corpus too small for meaningful TF-IDF. Need at least 5 texts.")
            return
        self.vectorizer.fit(texts)
        self.fitted = True
        logger.info(f"TF-IDF fitted on {len(texts)} texts. Vocabulary: {len(self.vectorizer.vocabulary_)} words")

    def get_top_words(self, text: str, topk: int = 20) -> list[dict]:
        """Get top-k TF-IDF words for a single document against the fitted corpus."""
        if not self.fitted:
            raise RuntimeError("Must call fit() before get_top_words()")

        vec = self.vectorizer.transform([text])
        feature_names = self.vectorizer.get_feature_names_out()
        scores = vec.toarray()[0]

        top_indices = scores.argsort()[::-1][:topk]
        return [
            {"word": feature_names[i], "score": round(float(scores[i]), 4)}
            for i in top_indices if scores[i] > 0
        ]

    def aggregate_for_window(self, texts: list[str], topk: int = 50) -> list[dict]:
        """
        Aggregate TF-IDF scores across all documents in a time window.
        Returns the globally most important words for that window.
        """
        if not self.fitted or not texts:
            return []

        matrix = self.vectorizer.transform(texts)
        # Sum TF-IDF scores across all docs and normalize by doc count
        summed = np.asarray(matrix.sum(axis=0)).flatten()
        avg = summed / len(texts)

        feature_names = self.vectorizer.get_feature_names_out()
        top_indices = avg.argsort()[::-1][:topk]

        return [
            {"word": feature_names[i], "score": round(float(avg[i]), 4)}
            for i in top_indices if avg[i] > 0
        ]
```

### 7.3 Validate in `notebooks/04_keyword_exploration.ipynb`

#### Cell 1 — Per-Thread Keywords

```python
import sys
sys.path.append('../src')
from pipeline.keywords import extract_keywords_combined

# Use a real thread body if available, or a sample text
sample_text = """
最近看了几部国产科幻电影，感觉和好莱坞的差距还是挺明显的。
不是说特效不够好，而是剧情逻辑和世界观构建上比较薄弱。
当然也有优秀的例子，比如流浪地球系列，但整体来说还需要时间积累。
希望国内的科幻创作者能多从原著小说中汲取灵感，而不是一味追求视觉奇观。
"""

keywords = extract_keywords_combined(sample_text, topk=10)
print("Keywords:")
for kw in keywords:
    print(f"  {kw['word']}: {kw['score']:.4f}")
```

**What to look for:** Words like `科幻电影`, `好莱坞`, `特效`, `剧情`, `世界观`, `流浪地球` should appear. Not `感觉`, `整体`, `当然` (too common/generic).

#### Cell 2 — Corpus TF-IDF on Multiple Threads

```python
from pipeline.keywords import CorpusTFIDF

# Simulate multiple threads (use real data if available from Phase 2)
texts = [
    "这部电影的特效真的很好看，剧情也不错，强烈推荐",
    "导演的镜头语言非常成熟，演员表演也很到位，国产电影的希望",
    "科幻设定很有创意，但是逻辑有些混乱，后半段节奏拖沓",
    "流浪地球之后终于有能看的国产科幻了，期待续集",
    "特效一般但剧情扎实，这才是好电影应该有的样子",
]

tfidf = CorpusTFIDF()
tfidf.fit(texts)

print("Top words across corpus:")
top = tfidf.aggregate_for_window(texts, topk=15)
for w in top:
    print(f"  {w['word']}: {w['score']:.4f}")
```

### 7.4 Phase 5 Checklist

- [ ] `keywords.py` created with TextRank, TF-IDF, and combined extractor
- [ ] `CorpusTFIDF` class functional
- [ ] Per-thread keywords are semantically meaningful (not stopwords or single chars)
- [ ] Combined extractor surfaces different words than either method alone
- [ ] Corpus TF-IDF aggregate works on 5+ texts

---

## 8. Phase 6 — Sentiment Analysis with Pretrained Chinese Models

**Goal:** Add sentiment classification to the pipeline using a pretrained Chinese BERT model. Understand why zero-shot sentiment differs from fine-tuned sentiment, and establish a baseline before we fine-tune in Phase 8.

### 8.1 Choosing a Pretrained Model

Several Chinese BERT variants exist, each with different training data and objectives:

| Model | Training Data | Strengths |
|---|---|---|
| `bert-base-chinese` | Wikipedia + BooksCorpus (translated) | Baseline, well-documented |
| `hfl/chinese-roberta-wwm-ext` | General Chinese web + Wikipedia | Better than BERT-base, whole-word masking |
| `hfl/chinese-macbert-base` | General Chinese web | Better at understanding masked words; good fine-tuning base |
| `IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment` | Sentiment-specific data | Pretrained for sentiment — best zero-shot baseline |

For zero-shot sentiment (no fine-tuning), use `IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment`. For fine-tuning in Phase 8, we will use `hfl/chinese-macbert-base` as the base.

### 8.2 Create `src/pipeline/sentiment.py`

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from loguru import logger
import os


class SentimentAnalyzer:
    """
    Chinese sentiment classifier.

    In pipeline_version='pretrained', uses Erlangshen zero-shot.
    In pipeline_version='finetuned', uses your fine-tuned MacBERT from Phase 8.
    """

    PRETRAINED_MODEL = "IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment"
    LABEL_MAP = {0: "negative", 1: "positive"}  # Adjust based on actual model output

    def __init__(self, model_path: str = None, device: str = "auto"):
        """
        Args:
            model_path: Path to fine-tuned model checkpoint (Phase 8+).
                        If None, uses the pretrained Erlangshen model.
            device: 'auto' selects GPU if available, else CPU.
        """
        if device == "auto":
            self.device = 0 if torch.cuda.is_available() else -1
        else:
            self.device = device

        model_to_load = model_path or self.PRETRAINED_MODEL

        logger.info(f"Loading sentiment model: {model_to_load} on device {self.device}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_to_load)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_to_load)

        # HuggingFace pipeline wraps tokenizer + model for easy batched inference
        self.pipeline = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            truncation=True,
            max_length=512,
            batch_size=32  # Process 32 texts at once — efficient for 5-min batches
        )

        logger.info("Sentiment model loaded.")

    def analyze(self, texts: list[str]) -> list[dict]:
        """
        Classify sentiment for a batch of texts.

        Args:
            texts: List of preprocessed Chinese text strings.
                   Empty or None strings are handled gracefully.

        Returns:
            List of {"label": str, "score": float} dicts.
            label: "positive", "negative", or "neutral"
            score: Confidence [0.0, 1.0]
        """
        if not texts:
            return []

        # Replace empty strings with a placeholder to avoid pipeline errors
        safe_texts = [t if t and t.strip() else "无内容" for t in texts]

        try:
            raw_results = self.pipeline(safe_texts)
        except Exception as e:
            logger.error(f"Sentiment pipeline error: {e}")
            return [{"label": "neutral", "score": 0.5}] * len(texts)

        results = []
        for r in raw_results:
            label_raw = r["label"].lower()
            score = r["score"]

            # Normalize label names across different models
            if "pos" in label_raw or label_raw == "1" or label_raw == "label_1":
                label = "positive"
            elif "neg" in label_raw or label_raw == "0" or label_raw == "label_0":
                label = "negative"
            else:
                label = "neutral"

            # Confidence below 0.6 is treated as neutral
            # This threshold prevents weak predictions from skewing the aggregate
            if score < 0.6:
                label = "neutral"

            results.append({"label": label, "score": round(score, 4)})

        return results

    def analyze_with_context(self, title: str, body: str) -> dict:
        """
        Analyze sentiment using both title and body.
        Title carries more signal per character than body in Douban posts.
        We weight it 2:1 over the body score.
        """
        combined_text = f"{title}。{body[:400]}"  # Truncate body to stay in 512 tokens
        result = self.analyze([combined_text])
        return result[0] if result else {"label": "neutral", "score": 0.5}
```

### 8.3 Validate in `notebooks/05_sentiment_baseline.ipynb`

#### Cell 1 — Model Download and Basic Test

```python
import sys
sys.path.append('../src')
from pipeline.sentiment import SentimentAnalyzer

analyzer = SentimentAnalyzer()

test_texts = [
    "这部电影真的太好看了，剧情感人，特效震撼，强烈推荐！",
    "剧情逻辑混乱，演员表演生硬，完全浪费了这么好的题材，非常失望",
    "还行吧，不算特别好，也不算特别差，就一般般",
    "今天去看了，感觉还可以，下次有机会再看第二遍",
]

results = analyzer.analyze(test_texts)
for text, result in zip(test_texts, results):
    print(f"[{result['label']:8s} {result['score']:.2f}] {text[:40]}...")
```

**What to look for:** First text classified as positive, second as negative, third and fourth as neutral. If the model misclassifies the obvious ones, the label normalization in `analyze()` may need adjustment for this specific model's output format. Print `raw_results` to inspect the raw labels.

#### Cell 2 — Benchmark Throughput

```python
import time

# Simulate a 5-minute batch: roughly 50-100 threads
batch = test_texts * 25  # 100 texts
start = time.time()
results = analyzer.analyze(batch)
elapsed = time.time() - start

print(f"Analyzed {len(batch)} texts in {elapsed:.2f}s")
print(f"Rate: {len(batch)/elapsed:.1f} texts/second")
```

**What to look for:** At least 20 texts/second on CPU. If slower, reduce `batch_size` in the pipeline. The 5-minute NLP pipeline must finish all work in under 5 minutes — sentiment analysis should take less than 30 seconds for a typical batch.

### 8.4 Phase 6 Checklist

- [ ] `sentiment.py` created
- [ ] Erlangshen model downloads without error
- [ ] Positive/negative/neutral classifications are correct on test cases
- [ ] Label normalization handles model-specific label formats
- [ ] 100 texts processed in <10 seconds on CPU
- [ ] `analyze_with_context()` produces same or better results than body alone

---

## 9. Phase 7 — Topic Modeling with BERTopic

**Goal:** Discover what topics are being discussed in Douban threads without manually defining categories. BERTopic uses sentence embeddings + clustering to find coherent topics automatically.

### 9.1 How BERTopic Works

BERTopic is a neural topic modeling approach. Unlike classical LDA (which uses word co-occurrence statistics), BERTopic uses semantic meaning:

1. **Embed** each document using a sentence transformer → dense vector representation of meaning
2. **Reduce dimensions** with UMAP (from 768 dims to ~5 dims for clustering)
3. **Cluster** with HDBSCAN (density-based, finds clusters of any shape, marks outliers as -1)
4. **Represent topics** by finding the words most distinctive to each cluster using a class-based TF-IDF (c-TF-IDF)

The result: topics that are semantically coherent (similar meaning documents cluster together) rather than just keyword-similar.

**For Chinese text**, BERTopic needs a Chinese-capable sentence transformer. We use `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, which handles Chinese natively and produces 384-dimensional embeddings.

### 9.2 Create `src/pipeline/topics.py`

```python
import numpy as np
from bertopic import BERTopic
from sentence_transformers import SentenceTransformer
from umap import UMAP
from hdbscan import HDBSCAN
from sklearn.feature_extraction.text import CountVectorizer
from loguru import logger
import pickle
import os
from .preprocessor import segment, normalize_text


class ChineseTopicModeler:
    """
    BERTopic wrapper for Chinese text.

    The model is fit on a corpus and then used to assign topics
    to new incoming threads. Topic assignments are updated
    periodically (not every 5 minutes) to maintain stability.

    Two modes:
    - fit_transform(texts): Build a new topic model from scratch
    - transform(texts): Assign topics using an existing fitted model
    """

    MODEL_PATH = "models/bertopic_model"

    def __init__(self, n_topics: int = "auto", min_topic_size: int = 10):
        """
        Args:
            n_topics: Number of topics. "auto" lets HDBSCAN decide.
                      Set a number (e.g., 20) for more control.
            min_topic_size: Minimum documents per topic.
                           Smaller = more granular topics, more noise.
                           Larger = fewer, broader topics.
        """
        self.n_topics = n_topics
        self.min_topic_size = min_topic_size
        self.model: BERTopic | None = None
        self._embedding_model = None

    def _get_embedding_model(self):
        """Lazy-load the embedding model so it is only downloaded when needed."""
        if self._embedding_model is None:
            logger.info("Loading multilingual sentence transformer...")
            self._embedding_model = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )
        return self._embedding_model

    def _build_model(self) -> BERTopic:
        """
        Construct the BERTopic model with Chinese-optimized components.

        Each component is configurable — understanding why each setting
        was chosen is more important than memorizing the values.
        """
        embedding_model = self._get_embedding_model()

        # UMAP: Dimensionality reduction
        # n_components: Target dimensions. 5 is empirically good for clustering.
        # n_neighbors: Controls local vs global structure. 15 is a good default.
        # metric: cosine is better than euclidean for embeddings
        umap_model = UMAP(
            n_components=5,
            n_neighbors=15,
            min_dist=0.0,   # Compact clusters — important for HDBSCAN
            metric="cosine",
            random_state=42
        )

        # HDBSCAN: Density-based clustering
        # min_cluster_size = min_topic_size: the smallest topic we care about
        # metric: euclidean works on the UMAP-reduced space
        # cluster_selection_method: 'eom' (excess of mass) tends to find
        #   more meaningful clusters than 'leaf' for text data
        hdbscan_model = HDBSCAN(
            min_cluster_size=self.min_topic_size,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True  # Required for approximate_predict on new docs
        )

        # CountVectorizer: Used for c-TF-IDF topic representation
        # We use a custom tokenizer so Chinese words are segmented correctly
        vectorizer_model = CountVectorizer(
            tokenizer=self._chinese_tokenizer,
            token_pattern=None,
            min_df=2,
            max_df=0.9
        )

        n_topics_arg = None if self.n_topics == "auto" else self.n_topics

        model = BERTopic(
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            vectorizer_model=vectorizer_model,
            nr_topics=n_topics_arg,
            verbose=True
        )

        return model

    def _chinese_tokenizer(self, text: str) -> list[str]:
        """Custom tokenizer for the CountVectorizer used in BERTopic."""
        normalized = normalize_text(text)
        return segment(normalized, remove_stopwords=True, min_word_len=2)

    def fit_transform(self, texts: list[str]) -> tuple[list[int], list[float]]:
        """
        Fit a new topic model on a corpus and assign topics.

        Call this once when you have accumulated enough threads (300+ recommended).
        After fitting, use transform() for new documents.

        Args:
            texts: List of preprocessed thread texts (normalized, not tokenized —
                   BERTopic handles tokenization internally through the vectorizer)

        Returns:
            (topics, probabilities) — topic ID per document and confidence score.
            Topic ID -1 means the document was classified as an outlier (no clear topic).
        """
        if len(texts) < 50:
            logger.warning(
                f"Only {len(texts)} texts. Topic modeling needs 50+ for meaningful results. "
                "Returning all as outliers."
            )
            return [-1] * len(texts), [0.0] * len(texts)

        self.model = self._build_model()
        logger.info(f"Fitting BERTopic on {len(texts)} texts...")

        topics, probs = self.model.fit_transform(texts)
        n_topics = len(set(topics)) - (1 if -1 in topics else 0)
        outlier_pct = topics.count(-1) / len(topics) * 100 if isinstance(topics, list) else (topics == -1).mean() * 100

        logger.info(f"Found {n_topics} topics. Outlier rate: {outlier_pct:.1f}%")

        self.save()
        return topics, probs

    def transform(self, texts: list[str]) -> tuple[list[int], list[float]]:
        """
        Assign topics to new documents using the fitted model.

        Args:
            texts: List of new thread texts to classify

        Returns:
            (topics, probabilities)
        """
        if self.model is None:
            logger.warning("No fitted model. Run fit_transform() first.")
            return [-1] * len(texts), [0.0] * len(texts)

        topics, probs = self.model.transform(texts)
        return topics, probs

    def get_topic_info(self) -> list[dict]:
        """
        Get human-readable topic descriptions.

        Returns:
            List of {"topic_id": int, "label": str, "top_words": list} dicts
        """
        if self.model is None:
            return []

        topic_info = self.model.get_topic_info()
        results = []

        for _, row in topic_info.iterrows():
            tid = row["Topic"]
            if tid == -1:
                continue  # Skip the outlier pseudo-topic

            # Get top 5 words for this topic
            top_words_raw = self.model.get_topic(tid)
            top_words = [w for w, _ in top_words_raw[:5]] if top_words_raw else []
            label = "_".join(top_words[:3])  # Human-readable: "电影_剧情_演员"

            results.append({
                "topic_id": tid,
                "label": label,
                "top_words": top_words,
                "count": int(row["Count"])
            })

        return results

    def save(self) -> None:
        """Save fitted model to disk."""
        os.makedirs("models", exist_ok=True)
        self.model.save(self.MODEL_PATH)
        logger.info(f"BERTopic model saved to {self.MODEL_PATH}")

    def load(self) -> bool:
        """Load a previously fitted model. Returns True if successful."""
        if not os.path.exists(self.MODEL_PATH):
            logger.info("No saved BERTopic model found.")
            return False
        try:
            self.model = BERTopic.load(self.MODEL_PATH)
            logger.info(f"BERTopic model loaded from {self.MODEL_PATH}")
            return True
        except Exception as e:
            logger.error(f"Failed to load BERTopic model: {e}")
            return False
```

### 9.3 Validate in `notebooks/06_bertopic_exploration.ipynb`

#### Cell 1 — Fit on Sample Corpus

```python
import sys
sys.path.append('../src')
from pipeline.topics import ChineseTopicModeler

# Use real scraped threads if available (Phase 2 data)
# Otherwise use a synthetic sample to verify the pipeline
sample_texts = [
    "这部国产科幻电影的特效真的很厉害，希望以后能看到更多这样的作品",
    "流浪地球2比第一部还好看，特效升级了很多，刘培强的故事也更丰富",
    "感觉国内科幻还是差好莱坞一大截，主要是编剧能力不够",
    "今天去图书馆发现了一本好书，读完之后感触很深，推荐给大家",
    "最近在看的一本小说写得太好了，故事情节引人入胜",
    "这家餐厅的川菜做得非常正宗，辣度刚好，价格也实惠",
    "附近新开了一家火锅店，汤底很鲜，肉质新鲜",
    "北京的冬天来得太快了，昨天还穿衬衫今天就要穿羽绒服",
    "今年冬天特别冷，供暖还没开始真的很难熬",
] * 10  # Repeat to get 90 texts for clustering

modeler = ChineseTopicModeler(min_topic_size=5)
topics, probs = modeler.fit_transform(sample_texts)

print("Topic distribution:", {t: topics.count(t) for t in set(topics)})
print("\nTopic info:")
for topic in modeler.get_topic_info():
    print(f"  Topic {topic['topic_id']}: {topic['label']} ({topic['count']} docs)")
```

**What to look for:** With this sample you should see 2–4 topics roughly corresponding to: sci-fi movies, books, food, weather. Outlier count (-1) should be below 30% of total texts.

### 9.4 Phase 7 Checklist

- [ ] `topics.py` created
- [ ] BERTopic and UMAP install without conflicts
- [ ] `fit_transform()` runs without error on 50+ texts
- [ ] Discovered topics are semantically coherent (not random word combinations)
- [ ] Outlier rate below 40%
- [ ] `save()` and `load()` work correctly
- [ ] `get_topic_info()` returns readable labels

---

## 10. Phase 8 — Fine-Tuning a Chinese BERT Model

**Goal:** Train a sentiment classifier specifically for Douban social text using your own scraped data. Understand what fine-tuning actually does to model weights and why domain-specific fine-tuning outperforms a generic pretrained model.

### 10.1 What Fine-Tuning Actually Does

A pretrained model like MacBERT has learned general Chinese language understanding from billions of words. But "understanding language" is not the same as "performing well on your specific task." Fine-tuning adapts the model's weights for your task through gradient descent — the same learning process as pretraining, but:

- Starting from already-useful weights (not random initialization)
- On a much smaller task-specific dataset (hundreds to thousands of examples)
- With a much lower learning rate (we don't want to destroy the pretrained knowledge)
- For a short number of steps (2–5 epochs typically)

The result: a model whose internal representations have been nudged from "general Chinese language" toward "Douban sentiment classification."

**Why MacBERT specifically?** MacBERT (MLM as Correction BERT) was pretrained with a better masking strategy than vanilla BERT — instead of masking with `[MASK]` tokens, it masks with similar words, making the model more robust to natural language variation. This makes it a strong base for fine-tuning on informal social text.

### 10.2 Prepare Training Data

You need labeled sentiment data. Options in order of preference:

**Option A: Label your own scraped data (best domain match)**
Use the pretrained Erlangshen model from Phase 6 to generate pseudo-labels for your scraped threads, then manually verify a subset. This is called **self-training** or **pseudo-labeling**.

**Option B: Public Chinese sentiment datasets**
- `ChnSentiCorp` — 12k hotel/book/computer reviews with positive/negative labels. Available on HuggingFace as `seamew/ChnSentiCorp`
- `WAIMAI-10k` — 10k food delivery reviews. More colloquial than ChnSentiCorp
- `DOUBAN` — if you can find a labeled Douban-specific dataset, prioritize it

For this tutorial we use ChnSentiCorp as the training dataset and validate on your scraped data.

### 10.3 Create `src/training/finetune_bert.py`

This script is designed to run on Kaggle (free T4 GPU). Upload it to a Kaggle notebook and run it there.

```python
"""
Fine-tune MacBERT for Chinese sentiment classification.

Designed to run on Kaggle free tier (T4 GPU, 16GB VRAM).
Estimated runtime: ~45 minutes for 3 epochs on ChnSentiCorp.

Upload this script to a Kaggle notebook. Install dependencies:
    !pip install transformers datasets accelerate
"""

import os
import numpy as np
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report

# ---- Configuration ----
MODEL_NAME = "hfl/chinese-macbert-base"
DATASET_NAME = "seamew/ChnSentiCorp"
OUTPUT_DIR = "./macbert_sentiment"
NUM_LABELS = 2          # positive, negative
MAX_LENGTH = 256        # Max token length. 256 covers >95% of Douban posts.
BATCH_SIZE = 32         # T4 has 16GB VRAM — 32 is safe with MacBERT-base
LEARNING_RATE = 2e-5    # Standard fine-tuning LR for BERT models
NUM_EPOCHS = 3          # 3 epochs is usually enough; more risks overfitting
WARMUP_RATIO = 0.1      # Warm up LR for first 10% of training steps

# ---- Device Setup ----
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# ---- Load Dataset ----
print(f"\nLoading dataset: {DATASET_NAME}")
dataset = load_dataset(DATASET_NAME)
print(f"Train: {len(dataset['train'])} | Test: {len(dataset['test'])}")
print(f"Sample: {dataset['train'][0]}")

# ---- Tokenizer ----
print(f"\nLoading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

def tokenize_function(examples):
    """
    Tokenize text into input_ids, attention_mask, token_type_ids.

    truncation=True: Truncate sequences longer than MAX_LENGTH
    padding=False: We use DataCollatorWithPadding for dynamic padding
                   (padding to longest in batch, not all to MAX_LENGTH).
                   Dynamic padding is more memory-efficient.
    """
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False
    )

print("Tokenizing dataset...")
tokenized = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]  # Remove original text column; keep label
)

# Rename 'label' to what Trainer expects
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch")

# ---- Model ----
print(f"\nLoading model: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=NUM_LABELS,
    id2label={0: "negative", 1: "positive"},
    label2id={"negative": 0, "positive": 1}
)
model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params:,}")
print(f"Trainable parameters: {trainable_params:,}")

# ---- Metrics ----
def compute_metrics(eval_pred):
    """Compute accuracy and F1 during evaluation."""
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1 = f1_score(labels, predictions, average="weighted")
    return {"accuracy": acc, "f1": f1}

# ---- Training Arguments ----
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE * 2,  # Eval uses less memory
    learning_rate=LEARNING_RATE,
    warmup_ratio=WARMUP_RATIO,
    weight_decay=0.01,          # L2 regularization — prevents overfitting
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=torch.cuda.is_available(),  # Mixed precision training — 2x faster on GPU
    logging_steps=50,
    report_to="none",           # Disable wandb/tensorboard for simplicity
    dataloader_num_workers=2,
)

# ---- Trainer ----
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer),
    compute_metrics=compute_metrics,
)

# ---- Train ----
print("\nStarting fine-tuning...")
print(f"Steps per epoch: {len(tokenized['train']) // BATCH_SIZE}")
print(f"Total steps: {len(tokenized['train']) // BATCH_SIZE * NUM_EPOCHS}")

train_result = trainer.train()
print(f"\nTraining complete.")
print(f"Total time: {train_result.metrics['train_runtime']:.0f}s")
print(f"Final train loss: {train_result.metrics['train_loss']:.4f}")

# ---- Evaluate ----
print("\nFinal evaluation on test set:")
eval_results = trainer.evaluate()
print(f"Accuracy: {eval_results['eval_accuracy']:.4f}")
print(f"F1 Score: {eval_results['eval_f1']:.4f}")

# Detailed classification report
predictions = trainer.predict(tokenized["test"])
pred_labels = np.argmax(predictions.predictions, axis=-1)
true_labels = predictions.label_ids
print("\nClassification Report:")
print(classification_report(true_labels, pred_labels, target_names=["negative", "positive"]))

# ---- Save ----
print(f"\nSaving model to {OUTPUT_DIR}/best_model")
trainer.save_model(f"{OUTPUT_DIR}/best_model")
tokenizer.save_pretrained(f"{OUTPUT_DIR}/best_model")
print("Done. Download the model directory from Kaggle → Output.")
```

### 10.4 Understanding Training Dynamics

After each epoch, watch these metrics:

**Training loss:** Should decrease monotonically. If it plateaus early, increase `num_epochs`. If it increases, you are overfitting — add dropout or reduce `num_epochs`.

**Eval accuracy/F1:** Should improve epoch over epoch. If eval F1 improves while train loss keeps dropping dramatically, you are overfitting. The gap between train loss and eval metrics is a diagnostic signal.

**Expected results on ChnSentiCorp:**
- Epoch 1: ~90% accuracy
- Epoch 2: ~92% accuracy
- Epoch 3: ~93% accuracy

These are standard benchmarks for MacBERT-base on ChnSentiCorp. If you see significantly lower numbers, check tokenization and label correctness.

### 10.5 Load the Fine-Tuned Model

After downloading the checkpoint from Kaggle, place it at `models/macbert_sentiment/` and update `sentiment.py`:

```python
# In your scheduler (Phase 11), initialize with the fine-tuned model:
analyzer = SentimentAnalyzer(model_path="models/macbert_sentiment/best_model")
```

### 10.6 Phase 8 Checklist

- [ ] ChnSentiCorp dataset loads without error
- [ ] MacBERT tokenizer and model load correctly
- [ ] Training runs for 3 epochs on Kaggle T4
- [ ] Eval F1 above 0.90 on ChnSentiCorp test set
- [ ] Classification report shows balanced performance across classes
- [ ] Model saved and downloaded from Kaggle
- [ ] `SentimentAnalyzer` updated to load fine-tuned checkpoint

---

## 11. Phase 9 — LoRA Fine-Tuning for Thread Summarization

**Goal:** Fine-tune a small Chinese language model to generate concise summaries of Douban threads using LoRA — a parameter-efficient method that trains a fraction of the weights rather than the full model.

### 11.1 What Is LoRA and Why Does It Exist?

Full fine-tuning (Phase 8) updates all model parameters. For a 110M-parameter MacBERT, this is manageable. For a 1.5B-parameter language model, full fine-tuning requires:
- Storing optimizer states for every parameter (AdamW needs 2 states per param) = ~18GB for 1.5B params
- GPU VRAM that free tiers cannot provide

LoRA (Low-Rank Adaptation) solves this by injecting small trainable matrices into the model's attention layers while keeping the original weights frozen. If a weight matrix `W` has shape (768, 768), LoRA adds two matrices `A` (768×8) and `B` (8×768) where 8 is the **rank** — a hyperparameter. The effective update is `W + A×B`, but only `A` and `B` are trained.

The math behind why this works: the weight updates during fine-tuning tend to be low-rank (most of the information in the update can be captured by a small number of dimensions). LoRA exploits this to train orders of magnitude fewer parameters.

For rank=8 applied to all attention matrices in Qwen2-1.5B, LoRA trains ~7M parameters instead of 1.5B — a 200x reduction.

### 11.2 Prepare Summarization Training Data

Summarization requires (thread_text, summary) pairs. Options:

**Option A: Use a large Chinese LLM to generate summaries**
GPT-4 or Qwen2-72B can generate high-quality summaries for your scraped threads. Generate ~500–1000 (thread, summary) pairs and use them as training data for the small model.

**Option B: Use an existing Chinese summarization dataset**
`LCSTS` (Large Scale Chinese Short Text Summarization) is available on HuggingFace. It's news-domain, not social-media, but provides structural training signal.

For this phase we build the pipeline and training structure. You will provide the actual training data from your scraped threads.

### 11.3 Create `src/training/finetune_lora.py`

```python
"""
LoRA fine-tuning of Qwen2-1.5B for Douban thread summarization.
Designed to run on Kaggle free tier (T4, 16GB VRAM).

The model takes a thread (title + body) and generates a 1-3 sentence
summary that captures the main point and key sentiments expressed.

Estimated VRAM usage: ~10GB (fits T4 with room to spare).
Estimated runtime: ~2 hours for 3 epochs on 1000 examples.
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from datasets import Dataset
import json

# ---- Configuration ----
BASE_MODEL = "Qwen/Qwen2-1.5B-Instruct"
OUTPUT_DIR = "./qwen2_lora_summarizer"
LORA_RANK = 8            # Rank of LoRA matrices. Higher = more capacity but more params.
LORA_ALPHA = 16          # Scaling factor: effective_lr = alpha/rank * lr. Common: alpha = 2*rank
LORA_DROPOUT = 0.05      # Dropout on LoRA layers — small regularization
MAX_INPUT_LENGTH = 512   # Max tokens for thread input
MAX_OUTPUT_LENGTH = 128  # Max tokens for summary output
BATCH_SIZE = 4           # Small batch because we have long sequences
GRAD_ACCUM = 8           # Gradient accumulation = effective batch of 32
LEARNING_RATE = 3e-4     # LoRA can use higher LR than full fine-tuning
NUM_EPOCHS = 3

# ---- Load Tokenizer ----
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---- Load Model ----
print("Loading base model...")
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,     # Load in fp16 to save memory
    device_map="auto",             # Distribute across available GPUs/CPU
    trust_remote_code=True
)

# Prepare model for LoRA
# This freezes all base model weights and sets them to non-gradient
model = prepare_model_for_kbit_training(model)

# ---- LoRA Configuration ----
# target_modules: which attention components to add LoRA to
# Qwen2 uses q_proj, k_proj, v_proj, o_proj — these are standard attention projections
# Adding LoRA to all four covers the full attention mechanism
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=LORA_RANK,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    bias="none"  # Don't add LoRA to bias terms
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# Expected output: ~0.5% of parameters trainable

# ---- Data Format ----
# Qwen2-Instruct uses a chat template. We format our training examples
# as instruction-following conversations.

SYSTEM_PROMPT = (
    "你是一个擅长总结豆瓣帖子的助手。"
    "请用1-3句话总结帖子的主要内容和关键观点。"
    "总结应简洁、客观，保留帖子的核心情感色彩。"
)

def format_example(thread_title: str, thread_body: str, summary: str) -> str:
    """
    Format a single (thread, summary) pair into the Qwen2 chat template.
    This format tells the model what role to play and what the task is.
    """
    user_content = f"请总结以下豆瓣帖子：\n\n标题：{thread_title}\n\n内容：{thread_body[:400]}"

    # Qwen2-Instruct chat format
    conversation = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{user_content}<|im_end|>\n"
        f"<|im_start|>assistant\n{summary}<|im_end|>"
    )
    return conversation

def tokenize_example(example: dict) -> dict:
    """
    Tokenize a formatted training example.

    For causal LM training, we tokenize the full conversation
    (input + expected output) and set labels = input_ids
    except for the input portion (which gets label -100,
    meaning 'ignore in loss calculation').

    We only want the model to learn to predict the summary,
    not to predict the question it just read.
    """
    text = format_example(
        example["title"],
        example["body"],
        example["summary"]
    )

    tokenized = tokenizer(
        text,
        max_length=MAX_INPUT_LENGTH + MAX_OUTPUT_LENGTH,
        truncation=True,
        padding=False,
        return_tensors=None
    )

    # Find where the assistant response starts
    # We do this by tokenizing just the input portion and noting the length
    input_text = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n"
        f"请总结以下豆瓣帖子：\n\n标题：{example['title']}\n\n内容：{example['body'][:400]}"
        f"<|im_end|>\n<|im_start|>assistant\n"
    )
    input_ids_only = tokenizer(input_text, add_special_tokens=False)["input_ids"]
    input_len = len(input_ids_only)

    # Set labels: -100 for input tokens (ignored in loss), actual IDs for output tokens
    labels = [-100] * min(input_len, len(tokenized["input_ids"])) + \
             tokenized["input_ids"][input_len:]
    tokenized["labels"] = labels

    return tokenized

# ---- Load Your Dataset ----
# Replace this with your actual (title, body, summary) pairs
# Format: list of {"title": str, "body": str, "summary": str} dicts
# Generate summaries with a large LLM (GPT-4, Qwen2-72B) on your scraped threads

# Example placeholder — replace with real data:
training_data = [
    {
        "title": "今年最好看的国产科幻电影",
        "body": "看完之后感觉国产科幻终于有了自己的风格，特效很精良，剧情也不拖沓...",
        "summary": "作者认为该国产科幻电影在特效和剧情上都有突出表现，标志着国产科幻的成熟。"
    },
    # Add your real data here
]

dataset = Dataset.from_list(training_data)
dataset = dataset.map(tokenize_example, remove_columns=dataset.column_names)
split = dataset.train_test_split(test_size=0.1, seed=42)

# ---- Training ----
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    fp16=True,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    logging_steps=20,
    warmup_steps=50,
    report_to="none"
)

from transformers import Trainer, DataCollatorForSeq2Seq

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=split["train"],
    eval_dataset=split["test"],
    data_collator=DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        padding=True,
        pad_to_multiple_of=8
    )
)

trainer.train()
trainer.save_model(f"{OUTPUT_DIR}/final")
print(f"LoRA adapter saved to {OUTPUT_DIR}/final")
```

### 11.4 Create `src/pipeline/summarizer.py`

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from loguru import logger
import os


class ThreadSummarizer:
    """
    Qwen2 LoRA summarizer for Douban threads.
    Falls back gracefully if model is not yet available.
    """

    BASE_MODEL = "Qwen/Qwen2-1.5B-Instruct"
    ADAPTER_PATH = "models/qwen2_lora_summarizer/final"
    SYSTEM_PROMPT = (
        "你是一个擅长总结豆瓣帖子的助手。"
        "请用1-3句话总结帖子的主要内容和关键观点。"
    )

    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.available = False
        self._try_load()

    def _try_load(self):
        """Load model if adapter exists. Skip silently if not yet trained."""
        if not os.path.exists(self.ADAPTER_PATH):
            logger.info(
                f"Summarizer adapter not found at {self.ADAPTER_PATH}. "
                "Summarization will be skipped until Phase 9 is complete."
            )
            return

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.BASE_MODEL, trust_remote_code=True
            )
            base = AutoModelForCausalLM.from_pretrained(
                self.BASE_MODEL,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            self.model = PeftModel.from_pretrained(base, self.ADAPTER_PATH)
            self.model.eval()
            self.available = True
            logger.info("Summarizer loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load summarizer: {e}")

    def summarize(self, title: str, body: str) -> str | None:
        """Generate a summary for a single thread. Returns None if model unavailable."""
        if not self.available:
            return None

        prompt = (
            f"<|im_start|>system\n{self.SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n请总结以下豆瓣帖子：\n\n标题：{title}\n\n内容：{body[:400]}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,        # Greedy decoding for deterministic summaries
                temperature=1.0,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # Decode only the newly generated tokens (not the input prompt)
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        summary = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return summary.strip()

    def summarize_batch(self, threads: list[dict]) -> list[str | None]:
        """Summarize a batch of threads. Each thread is {"title": str, "body": str}."""
        return [self.summarize(t["title"], t["body"]) for t in threads]
```

### 11.5 Phase 9 Checklist

- [ ] `finetune_lora.py` uploaded to Kaggle and runs without error
- [ ] `model.print_trainable_parameters()` shows ~0.5% trainable params
- [ ] Training loss decreases over 3 epochs
- [ ] LoRA adapter saved and downloaded from Kaggle
- [ ] `summarizer.py` created and loads adapter correctly
- [ ] `summarize()` produces coherent 1-3 sentence Chinese summaries
- [ ] Pipeline skips summarization gracefully when adapter not yet available

---

## 12. Phase 10 — Understanding What the Model Learned (Probing)

**Goal:** Gain empirical insight into what your fine-tuned BERT model actually encoded at each layer using probing classifiers. This is the phase that separates practitioners from engineers.

### 12.1 What Probing Tells You

A probing classifier is a simple model (usually logistic regression) trained to predict a specific property using the internal representations (hidden states) of a transformer at a specific layer.

If a simple classifier trained on layer 3's hidden states achieves 85% accuracy at predicting sentiment, that tells you: **layer 3 encodes sentiment information.** If layer 11 achieves 92%, layer 11 is more relevant for sentiment.

For Chinese social text, probing reveals:
- Which layers encode syntactic information (part-of-speech, word boundaries)
- Which layers encode semantic information (topic, sentiment)
- Whether your fine-tuned model shifted its representations relative to the base model
- Where in the network the domain adaptation happened

### 12.2 Create `src/training/probing.py`

```python
"""
Probing classifiers for fine-tuned MacBERT.

Extracts hidden states from each transformer layer and trains
a logistic regression classifier at each layer to predict sentiment.
Compares the representation quality of base vs. fine-tuned models.
"""

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from loguru import logger
from tqdm import tqdm


def extract_hidden_states(
    model_path: str,
    texts: list[str],
    labels: list[int],
    batch_size: int = 16,
    layers_to_probe: list[int] = None
) -> dict:
    """
    Extract hidden states from all transformer layers for a set of texts.

    Args:
        model_path: Path to model (either pretrained name or fine-tuned checkpoint)
        texts: List of text strings
        labels: Corresponding integer labels
        batch_size: Inference batch size
        layers_to_probe: Which layers to extract. None = all layers.

    Returns:
        dict mapping layer_idx → numpy array of shape (n_texts, hidden_size)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        output_hidden_states=True,  # Critical: tells the model to return all hidden states
        num_labels=2
    )
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    n_layers = model.config.num_hidden_layers + 1  # +1 for embedding layer (layer 0)
    if layers_to_probe is None:
        layers_to_probe = list(range(n_layers))

    # Initialize accumulators: layer_idx → list of [CLS] vectors
    layer_states = {i: [] for i in layers_to_probe}

    with torch.no_grad():
        for start in tqdm(range(0, len(texts), batch_size), desc="Extracting states"):
            batch_texts = texts[start:start + batch_size]

            encoded = tokenizer(
                batch_texts,
                truncation=True,
                max_length=256,
                padding=True,
                return_tensors="pt"
            ).to(device)

            outputs = model(**encoded)

            # outputs.hidden_states is a tuple of length n_layers
            # Each element has shape (batch_size, seq_len, hidden_size)
            # We take the [CLS] token representation (index 0) as the
            # document-level representation — this is standard for classification probing
            for layer_idx in layers_to_probe:
                cls_vectors = outputs.hidden_states[layer_idx][:, 0, :].cpu().numpy()
                layer_states[layer_idx].append(cls_vectors)

    # Stack all batches
    return {
        layer_idx: np.vstack(vectors)
        for layer_idx, vectors in layer_states.items()
    }


def probe_layers(
    hidden_states: dict,
    labels: list[int],
    cv_folds: int = 5
) -> dict:
    """
    Train a logistic regression probe at each layer and measure accuracy.

    Args:
        hidden_states: Output of extract_hidden_states()
        labels: True class labels for each text
        cv_folds: Number of cross-validation folds

    Returns:
        dict mapping layer_idx → mean cross-val accuracy
    """
    results = {}

    for layer_idx, states in sorted(hidden_states.items()):
        # Logistic regression with a small C (strong regularization)
        # The probe should be simple — we don't want it to memorize the data
        # We want to know what information is available in the representations
        probe = LogisticRegression(
            C=0.1,              # Small C = strong regularization
            max_iter=1000,
            solver="lbfgs",
            multi_class="auto"
        )

        scores = cross_val_score(probe, states, labels, cv=cv_folds, scoring="accuracy")
        mean_acc = scores.mean()
        std_acc = scores.std()

        results[layer_idx] = {"mean_accuracy": mean_acc, "std": std_acc}
        logger.info(f"Layer {layer_idx:2d}: accuracy = {mean_acc:.4f} ± {std_acc:.4f}")

    return results


def compare_base_vs_finetuned(
    base_model_path: str,
    finetuned_model_path: str,
    texts: list[str],
    labels: list[int]
) -> dict:
    """
    Run probing on both base and fine-tuned model and compare layer accuracies.

    This reveals how fine-tuning shifted representations.
    Layers where fine-tuned >> base = the fine-tuning signal was absorbed there.
    Layers where base ≈ fine-tuned = those layers were not significantly affected.
    """
    logger.info("Probing base model...")
    base_states = extract_hidden_states(base_model_path, texts, labels)
    base_results = probe_layers(base_states, labels)

    logger.info("Probing fine-tuned model...")
    ft_states = extract_hidden_states(finetuned_model_path, texts, labels)
    ft_results = probe_layers(ft_states, labels)

    comparison = {}
    for layer_idx in base_results:
        base_acc = base_results[layer_idx]["mean_accuracy"]
        ft_acc = ft_results[layer_idx]["mean_accuracy"]
        delta = ft_acc - base_acc
        comparison[layer_idx] = {
            "base_accuracy": base_acc,
            "finetuned_accuracy": ft_acc,
            "delta": delta
        }

    return comparison
```

### 12.3 Validate in `notebooks/08_probing_experiments.ipynb`

```python
import sys
sys.path.append('../src')
from training.probing import compare_base_vs_finetuned
import matplotlib.pyplot as plt
import numpy as np

# Use your test set (50-100 examples is enough for probing)
test_texts = [...]  # Load from ChnSentiCorp test set
test_labels = [...]

comparison = compare_base_vs_finetuned(
    base_model_path="hfl/chinese-macbert-base",
    finetuned_model_path="models/macbert_sentiment/best_model",
    texts=test_texts[:100],
    labels=test_labels[:100]
)

# Plot layer-by-layer accuracy comparison
layers = sorted(comparison.keys())
base_accs = [comparison[l]["base_accuracy"] for l in layers]
ft_accs = [comparison[l]["finetuned_accuracy"] for l in layers]
deltas = [comparison[l]["delta"] for l in layers]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

ax1.plot(layers, base_accs, 'b-o', label='Base MacBERT', linewidth=2)
ax1.plot(layers, ft_accs, 'r-o', label='Fine-tuned MacBERT', linewidth=2)
ax1.set_xlabel("Layer")
ax1.set_ylabel("Probing Accuracy")
ax1.set_title("Sentiment Information Per Layer: Base vs Fine-tuned")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.bar(layers, deltas, color=['green' if d > 0 else 'red' for d in deltas])
ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
ax2.set_xlabel("Layer")
ax2.set_ylabel("Accuracy Delta (fine-tuned − base)")
ax2.set_title("Where Fine-tuning Changed Representations")
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("probing_results.png", dpi=150)
plt.show()
```

**What to look for:**
- Base MacBERT likely shows increasing accuracy through layers, plateauing at higher layers
- Fine-tuned model should show notably higher accuracy in upper layers (9–12) — this is where the task-specific information was encoded
- The delta plot shows which layers were most affected by fine-tuning: upper layers should show the largest positive deltas

### 12.4 Phase 10 Checklist

- [ ] `probing.py` created
- [ ] Hidden states extracted successfully for both base and fine-tuned model
- [ ] Logistic regression probes trained at all 13 layers
- [ ] Layer accuracy plot generated
- [ ] Fine-tuned model shows higher accuracy in upper layers than base model
- [ ] Delta plot reveals where fine-tuning signal was absorbed

---

## 13. Phase 11 — The 5-Minute Orchestration Pipeline

**Goal:** Wire all pipeline components (scraper, preprocessor, keywords, sentiment, topics, summarizer, aggregator) into a single orchestrated cycle that runs every 5 minutes without overlap, without skipping steps on failure, and within the 5-minute window.

### 13.1 The Timing Constraint

The pipeline must complete within 5 minutes. A typical cycle processes 50–100 new threads. Here is the per-cycle time budget:

| Step | Estimated Time |
|---|---|
| Scrape 3 pages (75 threads) | 30–60s (rate-limited) |
| Preprocess all texts | <2s |
| Keyword extraction (75 threads) | ~5s |
| Sentiment (75 threads, CPU) | ~15s |
| Topic assignment (75 threads) | ~10s |
| Summarization (75 threads, optional) | ~90s (GPU) / skip on CPU |
| Aggregate + write to DB | ~5s |
| **Total (without summarization)** | **~100s** |
| **Total (with summarization, GPU)** | **~4 min** |

Summarization is the only step that risks exceeding budget. The pipeline runs it only when GPU is available; otherwise it marks summaries as null and fills them in a background batch process.

### 13.2 Create `src/pipeline/aggregator.py`

```python
from datetime import datetime, timedelta
import json
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
from loguru import logger


async def write_nlp_results(
    session: AsyncSession,
    thread_id: str,
    sentiment_label: str,
    sentiment_score: float,
    topic_id: int,
    topic_label: str,
    keywords: list[dict],
    summary: str | None,
    embedding_id: str | None,
    pipeline_version: str = "1.0"
) -> None:
    """Write NLP results for a single thread to the nlp_results table."""
    await session.execute(text("""
        INSERT INTO nlp_results
            (thread_id, sentiment_label, sentiment_score, topic_id,
             topic_label, keywords, summary, embedding_id, pipeline_version)
        VALUES
            (:thread_id, :sentiment_label, :sentiment_score, :topic_id,
             :topic_label, :keywords, :summary, :embedding_id, :pipeline_version)
        ON CONFLICT DO NOTHING
    """), {
        "thread_id": thread_id,
        "sentiment_label": sentiment_label,
        "sentiment_score": sentiment_score,
        "topic_id": topic_id,
        "topic_label": topic_label,
        "keywords": json.dumps(keywords, ensure_ascii=False),
        "summary": summary,
        "embedding_id": embedding_id,
        "pipeline_version": pipeline_version
    })


async def compute_and_write_aggregates(
    session: AsyncSession,
    group_id: str,
    bucket: datetime
) -> None:
    """
    Compute hourly aggregates from nlp_results and write to the timeseries tables.

    This function is the bridge between raw NLP results and the dashboard.
    It runs after every pipeline cycle and pre-computes what the dashboard
    needs, so the dashboard only queries pre-aggregated data.

    Args:
        group_id: The group to aggregate for
        bucket: Timestamp rounded to the hour
    """
    # Round bucket to the nearest hour
    hour_bucket = bucket.replace(minute=0, second=0, microsecond=0)

    # ---- Keyword aggregation ----
    # Find threads processed in this hour window and aggregate their keywords
    await session.execute(text("""
        INSERT INTO keyword_timeseries (bucket, group_id, word, frequency, avg_tfidf)
        SELECT
            :bucket AS bucket,
            t.group_id,
            kw->>'word' AS word,
            COUNT(*) AS frequency,
            AVG((kw->>'score')::float) AS avg_tfidf
        FROM nlp_results nr
        JOIN threads t ON t.thread_id = nr.thread_id
        CROSS JOIN LATERAL jsonb_array_elements(nr.keywords) AS kw
        WHERE
            t.group_id = :group_id
            AND nr.processed_at >= :hour_start
            AND nr.processed_at < :hour_end
        GROUP BY t.group_id, kw->>'word'
        ON CONFLICT (bucket, group_id, word) DO UPDATE SET
            frequency = EXCLUDED.frequency,
            avg_tfidf = EXCLUDED.avg_tfidf
    """), {
        "bucket": hour_bucket,
        "group_id": group_id,
        "hour_start": hour_bucket,
        "hour_end": hour_bucket + timedelta(hours=1)
    })

    # ---- Sentiment aggregation ----
    await session.execute(text("""
        INSERT INTO sentiment_timeseries
            (bucket, group_id, avg_score, positive_count, negative_count, neutral_count, total_count)
        SELECT
            :bucket,
            t.group_id,
            AVG(nr.sentiment_score),
            COUNT(*) FILTER (WHERE nr.sentiment_label = 'positive'),
            COUNT(*) FILTER (WHERE nr.sentiment_label = 'negative'),
            COUNT(*) FILTER (WHERE nr.sentiment_label = 'neutral'),
            COUNT(*)
        FROM nlp_results nr
        JOIN threads t ON t.thread_id = nr.thread_id
        WHERE
            t.group_id = :group_id
            AND nr.processed_at >= :hour_start
            AND nr.processed_at < :hour_end
        GROUP BY t.group_id
        ON CONFLICT (bucket, group_id) DO UPDATE SET
            avg_score = EXCLUDED.avg_score,
            positive_count = EXCLUDED.positive_count,
            negative_count = EXCLUDED.negative_count,
            neutral_count = EXCLUDED.neutral_count,
            total_count = EXCLUDED.total_count
    """), {
        "bucket": hour_bucket,
        "group_id": group_id,
        "hour_start": hour_bucket,
        "hour_end": hour_bucket + timedelta(hours=1)
    })

    # ---- Topic aggregation ----
    await session.execute(text("""
        INSERT INTO topic_timeseries (bucket, group_id, topic_id, topic_label, thread_count)
        SELECT
            :bucket,
            t.group_id,
            nr.topic_id,
            nr.topic_label,
            COUNT(*)
        FROM nlp_results nr
        JOIN threads t ON t.thread_id = nr.thread_id
        WHERE
            t.group_id = :group_id
            AND nr.topic_id != -1
            AND nr.processed_at >= :hour_start
            AND nr.processed_at < :hour_end
        GROUP BY t.group_id, nr.topic_id, nr.topic_label
        ON CONFLICT (bucket, group_id, topic_id) DO UPDATE SET
            thread_count = EXCLUDED.thread_count,
            topic_label = EXCLUDED.topic_label
    """), {
        "bucket": hour_bucket,
        "group_id": group_id,
        "hour_start": hour_bucket,
        "hour_end": hour_bucket + timedelta(hours=1)
    })

    logger.info(f"Aggregates written for group {group_id} at {hour_bucket}")
```

### 13.3 Create `src/scheduler.py`

```python
"""
5-minute orchestration pipeline.

Architecture:
    APScheduler fires every 5 minutes.
    Each cycle: scrape → store raw → NLP → aggregate.
    Overlap protection: if previous cycle not finished, current cycle skips.
    Error isolation: any step failure is logged but does not crash the scheduler.
"""

import asyncio
import os
import uuid
from datetime import datetime
from loguru import logger
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from dotenv import load_dotenv

load_dotenv()

# Pipeline components
from client.douban_client import DoubanClient
from db.connection import AsyncSessionLocal
from db.models import Thread as ThreadModel, Reply as ReplyModel, Group, NLPResult
from pipeline.preprocessor import preprocess
from pipeline.keywords import extract_keywords_combined
from pipeline.sentiment import SentimentAnalyzer
from pipeline.topics import ChineseTopicModeler
from pipeline.summarizer import ThreadSummarizer
from pipeline.aggregator import write_nlp_results, compute_and_write_aggregates
from sqlalchemy import text
from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct, Distance, VectorParams
from sentence_transformers import SentenceTransformer

PIPELINE_VERSION = "1.0"
GROUP_ID = os.getenv("TARGET_GROUP_ID", "")
QDRANT_COLLECTION = "thread_embeddings"
EMBEDDING_DIM = 384  # paraphrase-multilingual-MiniLM-L12-v2


class ObservatorPipeline:
    """
    Manages all long-lived resources and runs the 5-minute cycle.

    Resources (models, DB connections, HTTP clients) are initialized
    once at startup and reused across cycles — not recreated every 5 minutes.
    """

    def __init__(self):
        self.running = False
        self.cycle_count = 0

        # All resources initialized in setup()
        self.douban_client: DoubanClient | None = None
        self.sentiment_analyzer: SentimentAnalyzer | None = None
        self.topic_modeler: ChineseTopicModeler | None = None
        self.summarizer: ThreadSummarizer | None = None
        self.qdrant: QdrantClient | None = None
        self.embedding_model: SentenceTransformer | None = None

    async def setup(self):
        """Initialize all resources. Called once at startup."""
        logger.info("Initializing observatory pipeline...")

        # Scraping client
        self.douban_client = DoubanClient()
        await self.douban_client.initialize()

        # Sentiment: try fine-tuned first, fall back to pretrained
        finetuned_path = "models/macbert_sentiment/best_model"
        model_path = finetuned_path if os.path.exists(finetuned_path) else None
        self.sentiment_analyzer = SentimentAnalyzer(model_path=model_path)

        # Topic modeler: load saved model if exists
        self.topic_modeler = ChineseTopicModeler(min_topic_size=10)
        model_loaded = self.topic_modeler.load()
        if not model_loaded:
            logger.info(
                "No BERTopic model found. Topics will be assigned after "
                "300+ threads are accumulated and fit_transform() is called manually."
            )

        # Summarizer (optional — skips gracefully if LoRA not yet trained)
        self.summarizer = ThreadSummarizer()

        # Qdrant client and collection setup
        self.qdrant = QdrantClient(
            host=os.getenv("QDRANT_HOST", "localhost"),
            port=int(os.getenv("QDRANT_PORT", 6333))
        )
        self._setup_qdrant_collection()

        # Embedding model for Qdrant
        self.embedding_model = SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

        # Ensure target group exists in DB
        async with AsyncSessionLocal() as session:
            await session.execute(text("""
                INSERT INTO groups (group_id, name, last_scraped_at)
                VALUES (:group_id, :name, NOW())
                ON CONFLICT (group_id) DO NOTHING
            """), {"group_id": GROUP_ID, "name": f"Group {GROUP_ID}"})
            await session.commit()

        logger.info("Pipeline initialized and ready.")

    def _setup_qdrant_collection(self):
        """Create Qdrant collection if it doesn't exist."""
        collections = [c.name for c in self.qdrant.get_collections().collections]
        if QDRANT_COLLECTION not in collections:
            self.qdrant.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=EMBEDDING_DIM, distance=Distance.COSINE)
            )
            logger.info(f"Created Qdrant collection: {QDRANT_COLLECTION}")

    async def run_cycle(self):
        """
        Single 5-minute pipeline cycle.

        Steps:
        1. Scrape new threads from Douban
        2. Store raw threads and replies
        3. Run NLP on new threads only
        4. Write NLP results and Qdrant embeddings
        5. Recompute hourly aggregates
        """
        if self.running:
            logger.warning("Previous cycle still running. Skipping this cycle.")
            return

        self.running = True
        self.cycle_count += 1
        cycle_start = datetime.utcnow()
        logger.info(f"Cycle {self.cycle_count} starting at {cycle_start.isoformat()}")

        try:
            # ---- Step 1: Scrape ----
            threads = await self.douban_client.get_group_threads(GROUP_ID, max_pages=3)

            if not threads:
                logger.info("No threads fetched. Ending cycle early.")
                return

            # ---- Step 2: Store raw data ----
            new_thread_ids = await self._store_threads(threads)
            logger.info(f"Stored {len(new_thread_ids)} new threads out of {len(threads)} fetched")

            if not new_thread_ids:
                logger.info("No new threads to process.")
                return

            # Fetch full thread content for new threads only
            new_threads = [t for t in threads if t.thread_id in new_thread_ids]
            full_threads = []
            for thread in new_threads[:20]:  # Limit to 20 full fetches per cycle
                full = await self.douban_client.get_thread(thread.thread_id)
                if full:
                    full_threads.append(full)
                    await self._store_replies(full)

            # ---- Step 3: NLP ----
            nlp_inputs = [
                {"thread_id": t.thread_id, "title": t.title, "body": t.body or ""}
                for t in full_threads
            ]
            await self._run_nlp(nlp_inputs)

            # ---- Step 4: Aggregate ----
            await self._update_aggregates(GROUP_ID, cycle_start)

            elapsed = (datetime.utcnow() - cycle_start).total_seconds()
            logger.info(f"Cycle {self.cycle_count} complete in {elapsed:.1f}s")

        except Exception as e:
            logger.error(f"Cycle {self.cycle_count} failed: {e}", exc_info=True)
        finally:
            self.running = False

    async def _store_threads(self, threads) -> set[str]:
        """Insert new threads into DB. Returns set of newly inserted thread IDs."""
        new_ids = set()
        async with AsyncSessionLocal() as session:
            for thread in threads:
                result = await session.execute(text("""
                    INSERT INTO threads
                        (thread_id, group_id, title, author_id, author_name,
                         body, reply_count, url, created_at, last_active_at)
                    VALUES
                        (:thread_id, :group_id, :title, :author_id, :author_name,
                         :body, :reply_count, :url, :created_at, :last_active_at)
                    ON CONFLICT (thread_id) DO NOTHING
                    RETURNING thread_id
                """), {
                    "thread_id": thread.thread_id,
                    "group_id": thread.group_id or GROUP_ID,
                    "title": thread.title,
                    "author_id": thread.author_id,
                    "author_name": thread.author_name,
                    "body": thread.body,
                    "reply_count": thread.reply_count,
                    "url": thread.url,
                    "created_at": thread.created_at,
                    "last_active_at": thread.last_active_at,
                })
                row = result.fetchone()
                if row:
                    new_ids.add(row[0])
            await session.commit()
        return new_ids

    async def _store_replies(self, thread) -> None:
        """Insert replies for a thread."""
        if not thread.replies:
            return
        async with AsyncSessionLocal() as session:
            for reply in thread.replies:
                await session.execute(text("""
                    INSERT INTO replies
                        (reply_id, thread_id, author_id, author_name, body, created_at)
                    VALUES
                        (:reply_id, :thread_id, :author_id, :author_name, :body, :created_at)
                    ON CONFLICT (reply_id) DO NOTHING
                """), {
                    "reply_id": reply.reply_id,
                    "thread_id": reply.thread_id,
                    "author_id": reply.author_id,
                    "author_name": reply.author_name,
                    "body": reply.body,
                    "created_at": reply.created_at,
                })
            await session.commit()

    async def _run_nlp(self, nlp_inputs: list[dict]) -> None:
        """
        Run the full NLP pipeline on a batch of threads.
        All models are called in batch for efficiency.
        """
        if not nlp_inputs:
            return

        # Preprocess all texts
        preprocessed = [preprocess(item["body"]) for item in nlp_inputs]
        normalized_texts = [p["normalized"] for p in preprocessed]

        # Keyword extraction (per-thread)
        all_keywords = [
            extract_keywords_combined(text, topk=10)
            for text in normalized_texts
        ]

        # Sentiment (batched)
        full_texts = [
            f"{item['title']}。{item['body'][:400]}"
            for item in nlp_inputs
        ]
        sentiment_results = self.sentiment_analyzer.analyze(full_texts)

        # Topic assignment (batched, only if model is available)
        if self.topic_modeler.model is not None:
            topics, _ = self.topic_modeler.transform(normalized_texts)
            topic_info = {t["topic_id"]: t["label"] for t in self.topic_modeler.get_topic_info()}
        else:
            topics = [-1] * len(nlp_inputs)
            topic_info = {}

        # Embeddings for Qdrant
        embeddings = self.embedding_model.encode(
            normalized_texts,
            batch_size=32,
            show_progress_bar=False
        )

        # Summarization (optional, only if model loaded)
        if self.summarizer.available:
            summaries = self.summarizer.summarize_batch(
                [{"title": item["title"], "body": item["body"]} for item in nlp_inputs]
            )
        else:
            summaries = [None] * len(nlp_inputs)

        # Write everything to DB and Qdrant
        async with AsyncSessionLocal() as session:
            for i, item in enumerate(nlp_inputs):
                # Write to Qdrant
                embedding_id = str(uuid.uuid4())
                self.qdrant.upsert(
                    collection_name=QDRANT_COLLECTION,
                    points=[PointStruct(
                        id=embedding_id,
                        vector=embeddings[i].tolist(),
                        payload={
                            "thread_id": item["thread_id"],
                            "group_id": GROUP_ID,
                            "title": item["title"]
                        }
                    )]
                )

                # Write NLP results to Postgres
                await write_nlp_results(
                    session=session,
                    thread_id=item["thread_id"],
                    sentiment_label=sentiment_results[i]["label"],
                    sentiment_score=sentiment_results[i]["score"],
                    topic_id=int(topics[i]),
                    topic_label=topic_info.get(int(topics[i]), "unknown"),
                    keywords=all_keywords[i],
                    summary=summaries[i],
                    embedding_id=embedding_id,
                    pipeline_version=PIPELINE_VERSION
                )

            await session.commit()

        logger.info(f"NLP complete for {len(nlp_inputs)} threads.")

    async def _update_aggregates(self, group_id: str, bucket: datetime) -> None:
        """Recompute hourly aggregates for the current time window."""
        async with AsyncSessionLocal() as session:
            await compute_and_write_aggregates(session, group_id, bucket)
            await session.commit()

    async def teardown(self):
        """Clean up resources on shutdown."""
        if self.douban_client:
            await self.douban_client.close()


# ---- Main Entry Point ----
async def main():
    pipeline = ObservatorPipeline()
    await pipeline.setup()

    scheduler = AsyncIOScheduler()
    scheduler.add_job(
        pipeline.run_cycle,
        trigger="interval",
        seconds=int(os.getenv("SCRAPE_INTERVAL", 300)),
        id="pipeline_cycle",
        max_instances=1,    # CRITICAL: prevents overlap — if a cycle is running,
                            # the next scheduled run is skipped, not queued
        coalesce=True       # If multiple missed fires accumulate (e.g., after a crash),
                            # only run once, not for each missed interval
    )

    scheduler.start()
    logger.info(f"Scheduler started. Cycle interval: {os.getenv('SCRAPE_INTERVAL', 300)}s")

    # Run one cycle immediately on start
    await pipeline.run_cycle()

    try:
        while True:
            await asyncio.sleep(60)  # Keep the event loop alive
    except (KeyboardInterrupt, SystemExit):
        logger.info("Shutdown signal received.")
        scheduler.shutdown()
        await pipeline.teardown()


if __name__ == "__main__":
    asyncio.run(main())
```

### 13.4 Phase 11 Checklist

- [ ] `aggregator.py` created with SQL aggregate queries
- [ ] `scheduler.py` created
- [ ] `max_instances=1` confirmed (overlap protection)
- [ ] Pipeline runs for at least 2 full cycles without error
- [ ] New threads appear in `threads` table after first cycle
- [ ] `nlp_results` table has rows after first cycle
- [ ] `keyword_timeseries` and `sentiment_timeseries` have rows after aggregation
- [ ] Qdrant collection shows points after first cycle
- [ ] Cycle completes in under 5 minutes (monitor the elapsed log line)

---

## 14. Phase 12 — Streamlit Time-Series Dashboard

**Goal:** Build the visualization layer. The dashboard displays how language, sentiment, and topics change over time — queryable at minute, hour, day, and month granularity for up to 1 year.

### 14.1 Architecture of the Dashboard

The dashboard never reads raw tables. It only queries:
- `keyword_timeseries` — for lexical clouds and keyword trend charts
- `sentiment_timeseries` — for sentiment trajectory charts
- `topic_timeseries` — for topic heatmaps
- `threads` + `nlp_results` (joined) — for the thread browser

All queries are time-windowed using TimescaleDB's `time_bucket()` function, which rounds timestamps to any interval you specify.

### 14.2 Build `app.py`

```python
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import asyncio
import asyncpg
import os
from wordcloud import WordCloud
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="豆瓣 NLP 观察站",
    page_icon="🔭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- Database Connection ----
@st.cache_resource
def get_db_connection_string():
    return (
        f"postgresql://{os.getenv('POSTGRES_USER')}:{os.getenv('POSTGRES_PASSWORD')}"
        f"@{os.getenv('POSTGRES_HOST')}:{os.getenv('POSTGRES_PORT')}"
        f"/{os.getenv('POSTGRES_DB')}"
    )

@st.cache_data(ttl=60)  # Cache queries for 60 seconds — dashboard refreshes every minute
def query_db(sql: str, params: dict = None) -> pd.DataFrame:
    """
    Run a SQL query and return results as a DataFrame.
    Cached for 60 seconds to prevent hammering the DB on every interaction.
    """
    import psycopg2
    import psycopg2.extras

    conn_str = get_db_connection_string()
    conn = psycopg2.connect(conn_str)
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute(sql, params or {})
            rows = cur.fetchall()
        return pd.DataFrame(rows)
    finally:
        conn.close()


# ---- Sidebar Controls ----
with st.sidebar:
    st.title("🔭 观察站控制")

    # Time range selector
    st.subheader("时间范围")
    time_range = st.selectbox(
        "查看范围",
        options=["最近1小时", "最近6小时", "最近24小时", "最近7天", "最近30天", "最近1年"],
        index=2
    )

    time_range_map = {
        "最近1小时": (timedelta(hours=1), "1 minute"),
        "最近6小时": (timedelta(hours=6), "5 minutes"),
        "最近24小时": (timedelta(hours=24), "1 hour"),
        "最近7天": (timedelta(days=7), "6 hours"),
        "最近30天": (timedelta(days=30), "1 day"),
        "最近1年": (timedelta(days=365), "1 week"),
    }

    delta, bucket_interval = time_range_map[time_range]
    start_time = datetime.utcnow() - delta

    # Group selector
    groups_df = query_db("SELECT group_id, name FROM groups ORDER BY last_scraped_at DESC NULLS LAST")
    if groups_df.empty:
        st.warning("No groups scraped yet. Start the scheduler first.")
        st.stop()

    group_id = st.selectbox(
        "豆瓣小组",
        options=groups_df["group_id"].tolist(),
        format_func=lambda gid: groups_df[groups_df["group_id"] == gid]["name"].values[0]
    )

    # Keyword filter
    st.subheader("关键词筛选")
    keyword_filter = st.text_input("过滤关键词（可选）", placeholder="e.g. 电影")

    # Auto-refresh
    auto_refresh = st.toggle("自动刷新 (60s)", value=True)
    if auto_refresh:
        st.markdown(
            '<meta http-equiv="refresh" content="60">',
            unsafe_allow_html=True
        )

    st.markdown("---")

    # Pipeline status
    latest_log = query_db("""
        SELECT started_at, threads_new, error
        FROM scrape_log
        WHERE group_id = %(group_id)s
        ORDER BY started_at DESC
        LIMIT 1
    """, {"group_id": group_id})

    if not latest_log.empty:
        row = latest_log.iloc[0]
        status = "🟢 正常" if row["error"] is None else f"🔴 错误"
        st.metric("流水线状态", status)
        st.caption(f"最后更新: {row['started_at']}")


# ---- Main Dashboard ----
st.title("🔭 豆瓣 NLP 观察站")
st.caption(f"小组 {group_id} · {time_range} · 数据桶: {bucket_interval}")

# ---- Row 1: Summary Metrics ----
col1, col2, col3, col4 = st.columns(4)

metrics_df = query_db("""
    SELECT
        COUNT(DISTINCT t.thread_id) AS total_threads,
        COALESCE(SUM(st.total_count), 0) AS processed_threads,
        COALESCE(AVG(st.avg_score), 0.5) AS avg_sentiment,
        COALESCE(
            SUM(st.positive_count) * 100.0 / NULLIF(SUM(st.total_count), 0),
            50
        ) AS positive_pct
    FROM threads t
    LEFT JOIN sentiment_timeseries st ON st.group_id = t.group_id
        AND st.bucket >= %(start_time)s
    WHERE t.group_id = %(group_id)s
        AND t.scraped_at >= %(start_time)s
""", {"group_id": group_id, "start_time": start_time})

if not metrics_df.empty:
    m = metrics_df.iloc[0]
    with col1:
        st.metric("新帖数量", int(m["total_threads"]))
    with col2:
        st.metric("已处理", int(m["processed_threads"]))
    with col3:
        sentiment_val = float(m["avg_sentiment"])
        sentiment_emoji = "😊" if sentiment_val > 0.6 else ("😢" if sentiment_val < 0.4 else "😐")
        st.metric("平均情感", f"{sentiment_emoji} {sentiment_val:.2f}")
    with col4:
        st.metric("正向比例", f"{float(m['positive_pct']):.1f}%")

st.divider()

# ---- Row 2: Lexical Cloud + Keyword Trend ----
col_left, col_right = st.columns([1, 2])

with col_left:
    st.subheader("🔤 词语云图")

    keyword_cloud_df = query_db("""
        SELECT word, SUM(frequency) AS total_freq, AVG(avg_tfidf) AS avg_score
        FROM keyword_timeseries
        WHERE group_id = %(group_id)s
            AND bucket >= %(start_time)s
        GROUP BY word
        ORDER BY avg_score DESC
        LIMIT 100
    """, {"group_id": group_id, "start_time": start_time})

    if not keyword_cloud_df.empty:
        word_freq = dict(zip(keyword_cloud_df["word"], keyword_cloud_df["avg_score"]))

        # Generate wordcloud with Chinese font
        # Download a Chinese font: https://github.com/StellarCN/scp_zh/raw/master/fonts/simhei.ttf
        # Place it at: data/simhei.ttf
        font_path = "data/simhei.ttf"
        wc = WordCloud(
            font_path=font_path if os.path.exists(font_path) else None,
            width=400,
            height=300,
            background_color="white",
            max_words=80,
            colormap="viridis",
            prefer_horizontal=0.9
        ).generate_from_frequencies(word_freq)

        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.imshow(wc, interpolation="bilinear")
        ax.axis("off")
        st.pyplot(fig, use_container_width=True)
        plt.close()
    else:
        st.info("暂无关键词数据。等待第一个流水线周期完成。")

with col_right:
    st.subheader("📈 关键词趋势")

    # Get top 8 keywords and their time series
    top_keywords_df = query_db("""
        SELECT word, SUM(frequency) AS total
        FROM keyword_timeseries
        WHERE group_id = %(group_id)s
            AND bucket >= %(start_time)s
        GROUP BY word
        ORDER BY total DESC
        LIMIT 8
    """, {"group_id": group_id, "start_time": start_time})

    if not top_keywords_df.empty:
        top_words = top_keywords_df["word"].tolist()
        if keyword_filter:
            top_words = [w for w in top_words if keyword_filter in w] or top_words[:5]

        trend_df = query_db(f"""
            SELECT
                time_bucket('{bucket_interval}'::interval, bucket) AS time_bucket,
                word,
                SUM(frequency) AS frequency
            FROM keyword_timeseries
            WHERE group_id = %(group_id)s
                AND bucket >= %(start_time)s
                AND word = ANY(%(words)s)
            GROUP BY 1, 2
            ORDER BY 1
        """, {
            "group_id": group_id,
            "start_time": start_time,
            "words": top_words
        })

        if not trend_df.empty:
            fig = px.line(
                trend_df,
                x="time_bucket",
                y="frequency",
                color="word",
                title=f"关键词频率趋势 ({bucket_interval} 粒度)",
                labels={"time_bucket": "时间", "frequency": "出现次数", "word": "关键词"}
            )
            fig.update_layout(
                legend_title="关键词",
                hovermode="x unified",
                height=300
            )
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("暂无趋势数据。")

st.divider()

# ---- Row 3: Sentiment Time Series ----
st.subheader("💭 情感走势")

sentiment_df = query_db(f"""
    SELECT
        time_bucket('{bucket_interval}'::interval, bucket) AS time_bucket,
        AVG(avg_score) AS avg_score,
        SUM(positive_count) AS positive,
        SUM(negative_count) AS negative,
        SUM(neutral_count) AS neutral,
        SUM(total_count) AS total
    FROM sentiment_timeseries
    WHERE group_id = %(group_id)s
        AND bucket >= %(start_time)s
    GROUP BY 1
    ORDER BY 1
""", {"group_id": group_id, "start_time": start_time})

if not sentiment_df.empty:
    tab1, tab2 = st.tabs(["情感分数走势", "情感分布堆叠图"])

    with tab1:
        fig = px.line(
            sentiment_df,
            x="time_bucket",
            y="avg_score",
            title="平均情感分数走势",
            labels={"time_bucket": "时间", "avg_score": "情感分数 (0=负向, 1=正向)"},
        )
        fig.add_hline(y=0.5, line_dash="dash", line_color="gray", annotation_text="中性基线")
        fig.update_layout(height=300, hovermode="x")
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=sentiment_df["time_bucket"], y=sentiment_df["positive"],
            name="正向", marker_color="#2ecc71"
        ))
        fig.add_trace(go.Bar(
            x=sentiment_df["time_bucket"], y=sentiment_df["neutral"],
            name="中性", marker_color="#95a5a6"
        ))
        fig.add_trace(go.Bar(
            x=sentiment_df["time_bucket"], y=sentiment_df["negative"],
            name="负向", marker_color="#e74c3c"
        ))
        fig.update_layout(
            barmode="stack",
            title="情感分布堆叠图",
            xaxis_title="时间",
            yaxis_title="帖子数量",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)
else:
    st.info("暂无情感数据。")

st.divider()

# ---- Row 4: Topic Heatmap ----
st.subheader("🗂️ 话题热力图")

topic_df = query_db(f"""
    SELECT
        time_bucket('{bucket_interval}'::interval, bucket) AS time_bucket,
        topic_label,
        SUM(thread_count) AS count
    FROM topic_timeseries
    WHERE group_id = %(group_id)s
        AND bucket >= %(start_time)s
    GROUP BY 1, 2
    ORDER BY 1
""", {"group_id": group_id, "start_time": start_time})

if not topic_df.empty and len(topic_df["topic_label"].unique()) > 1:
    pivot = topic_df.pivot_table(
        index="topic_label",
        columns="time_bucket",
        values="count",
        fill_value=0
    )

    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="YlOrRd",
        title="话题随时间分布热力图",
        labels={"x": "时间", "y": "话题", "color": "帖子数"}
    )
    fig.update_layout(height=350)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("暂无话题数据。BERTopic 需要 300+ 帖子才能拟合。")

st.divider()

# ---- Row 5: Thread Browser ----
st.subheader("📋 帖子浏览器")

thread_df = query_db("""
    SELECT
        t.thread_id,
        t.title,
        t.author_name,
        t.reply_count,
        t.created_at,
        nr.sentiment_label,
        nr.sentiment_score,
        nr.topic_label,
        nr.summary,
        nr.keywords
    FROM threads t
    LEFT JOIN nlp_results nr ON nr.thread_id = t.thread_id
    WHERE t.group_id = %(group_id)s
        AND t.scraped_at >= %(start_time)s
    ORDER BY t.created_at DESC
    LIMIT 50
""", {"group_id": group_id, "start_time": start_time})

if not thread_df.empty:
    # Sentiment color coding
    def sentiment_badge(label):
        colors = {"positive": "🟢", "negative": "🔴", "neutral": "⚪"}
        return colors.get(label, "⚪")

    for _, row in thread_df.iterrows():
        badge = sentiment_badge(row.get("sentiment_label"))
        with st.expander(f"{badge} {row['title']} — {row['author_name']} ({row['reply_count']} 回复)"):
            col1, col2, col3 = st.columns(3)
            with col1:
                st.caption(f"发帖时间: {row['created_at']}")
                st.caption(f"情感: {row.get('sentiment_label', 'N/A')} ({row.get('sentiment_score', 0):.2f})")
            with col2:
                st.caption(f"话题: {row.get('topic_label', 'N/A')}")
            with col3:
                thread_url = f"https://www.douban.com/group/topic/{row['thread_id']}/"
                st.markdown(f"[查看原帖]({thread_url})")

            # Keywords
            if row.get("keywords"):
                import json
                try:
                    kws = json.loads(row["keywords"]) if isinstance(row["keywords"], str) else row["keywords"]
                    kw_str = " | ".join([f"`{k['word']}`" for k in kws[:8]])
                    st.markdown(f"**关键词:** {kw_str}")
                except Exception:
                    pass

            # AI Summary
            if row.get("summary"):
                st.markdown(f"**AI摘要:** {row['summary']}")
else:
    st.info("暂无帖子数据。")
```

### 14.3 Download a Chinese Font

The wordcloud requires a Chinese font to render characters correctly.

```bash
curl -L "https://github.com/StellarCN/scp_zh/raw/master/fonts/simhei.ttf" \
  -o data/simhei.ttf
```

Verify the file:
```bash
ls -lh data/simhei.ttf
# Should be ~9MB
```

### 14.4 Run the Dashboard

```bash
streamlit run app.py
```

Open `http://localhost:8501`.

### 14.5 Phase 12 Checklist

- [ ] `app.py` created and launches without import errors
- [ ] Dashboard loads without database errors
- [ ] Time range selector changes the data in all charts correctly
- [ ] Lexical cloud renders Chinese characters (requires simhei.ttf)
- [ ] Keyword trend chart shows multiple lines after 2+ hours of data
- [ ] Sentiment time series chart shows data after pipeline runs
- [ ] Thread browser shows threads with sentiment labels
- [ ] `@st.cache_data(ttl=60)` confirmed — not requerying every interaction
- [ ] Auto-refresh toggling works

---

## 15. Phase 13 — Tokenizer Training on Your Own Corpus

**Goal:** Train a custom Chinese tokenizer on your accumulated Douban corpus, giving it domain-specific vocabulary that standard tokenizers miss. This phase covers what tokenizers actually do and why domain vocabulary matters for model quality.

### 15.1 What a Tokenizer Does

Every transformer model converts text into a sequence of token IDs before processing. The tokenizer defines the vocabulary: which character sequences map to which IDs.

Standard Chinese BERT tokenizers use character-level tokenization (each Chinese character becomes one token). This is safe and general but has a drawback: compound words and domain-specific multi-character expressions are split across tokens.

For example, `流浪地球` (The Wandering Earth) in a character-level tokenizer becomes `['流', '浪', '地', '球']` — 4 tokens. A domain-aware tokenizer would treat it as a single token `['流浪地球']`, better capturing its meaning as a unit.

Training a tokenizer on your Douban corpus creates a vocabulary that contains high-frequency domain-specific terms as single tokens. This improves embedding quality for those terms.

### 15.2 Create `src/training/tokenizer_train.py`

```python
"""
Train a BPE tokenizer on your accumulated Douban corpus.

BPE (Byte-Pair Encoding) is the tokenization algorithm used by most modern
language models. It starts with a character vocabulary and iteratively merges
the most frequent character pairs into new vocabulary entries.

After training, you can use this tokenizer to:
1. Analyze your corpus vocabulary
2. Initialize a tokenizer for a custom model
3. Replace jieba with a data-driven segmentation for the NLP pipeline
"""

import os
import asyncio
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace, CharDelimiterSplit
from loguru import logger


async def export_corpus_from_db(output_path: str, min_length: int = 20) -> int:
    """
    Export all thread and reply text from Postgres to a plain text file.
    One document per line. This is the training corpus for the tokenizer.
    """
    import asyncpg
    from dotenv import load_dotenv
    load_dotenv()

    conn = await asyncpg.connect(
        host=os.getenv("POSTGRES_HOST"),
        port=int(os.getenv("POSTGRES_PORT")),
        database=os.getenv("POSTGRES_DB"),
        user=os.getenv("POSTGRES_USER"),
        password=os.getenv("POSTGRES_PASSWORD"),
    )

    count = 0
    with open(output_path, "w", encoding="utf-8") as f:
        # Threads
        rows = await conn.fetch(
            "SELECT title || '。' || COALESCE(body, '') AS text FROM threads "
            "WHERE LENGTH(body) >= $1", min_length
        )
        for row in rows:
            text = row["text"].replace("\n", " ").strip()
            if text:
                f.write(text + "\n")
                count += 1

        # Replies
        rows = await conn.fetch(
            "SELECT body AS text FROM replies WHERE LENGTH(body) >= $1", min_length
        )
        for row in rows:
            text = row["text"].replace("\n", " ").strip()
            if text:
                f.write(text + "\n")
                count += 1

    await conn.close()
    logger.info(f"Exported {count} documents to {output_path}")
    return count


def train_bpe_tokenizer(
    corpus_path: str,
    output_dir: str,
    vocab_size: int = 8000
) -> None:
    """
    Train a BPE tokenizer on the exported corpus.

    Args:
        corpus_path: Path to plain text file (one document per line)
        output_dir: Where to save the trained tokenizer
        vocab_size: Target vocabulary size.
                    8000 is reasonable for a domain-specific Chinese tokenizer.
                    Standard BERT-chinese has 21128 vocab entries.
    """
    logger.info(f"Training BPE tokenizer on {corpus_path} (vocab_size={vocab_size})")

    # Initialize empty BPE tokenizer
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))

    # Pre-tokenizer: Split on whitespace first
    # This is important for Chinese: we first segment with spaces (can use jieba output),
    # then BPE learns sub-word merges within those segments
    tokenizer.pre_tokenizer = Whitespace()

    # Special tokens — these must match what your downstream models expect
    special_tokens = ["[UNK]", "[CLS]", "[SEP]", "[PAD]", "[MASK]"]

    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=special_tokens,
        min_frequency=2,            # A token must appear at least twice to be included
        show_progress=True,
        initial_alphabet=list(
            "abcdefghijklmnopqrstuvwxyz"
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            "0123456789"
        )
    )

    # Train
    tokenizer.train([corpus_path], trainer)

    # Save
    os.makedirs(output_dir, exist_ok=True)
    tokenizer.save(os.path.join(output_dir, "tokenizer.json"))
    logger.info(f"Tokenizer saved to {output_dir}")

    # Report vocabulary statistics
    vocab = tokenizer.get_vocab()
    logger.info(f"Final vocabulary size: {len(vocab)}")

    # Show the top merges (most frequent character pair combinations)
    print("\nSample of learned merges (domain vocabulary):")
    for token, idx in sorted(vocab.items(), key=lambda x: x[1]):
        if len(token) > 1 and idx > len(special_tokens) + 100:  # Skip single chars and special tokens
            print(f"  '{token}' (id: {idx})")
            if idx > len(special_tokens) + 130:
                print("  ...")
                break


def analyze_tokenizer_coverage(tokenizer_path: str, test_texts: list[str]) -> None:
    """
    Compare how the domain tokenizer segments text vs. character-level BERT.
    Shows where domain vocabulary adds value.
    """
    from tokenizers import Tokenizer as HFTokenizer
    from transformers import AutoTokenizer

    domain_tokenizer = HFTokenizer.from_file(tokenizer_path)
    bert_tokenizer = AutoTokenizer.from_pretrained("hfl/chinese-macbert-base")

    print("\nTokenization comparison: Domain BPE vs MacBERT (character-level)")
    print("-" * 60)

    for text in test_texts:
        domain_tokens = domain_tokenizer.encode(text).tokens
        bert_tokens = bert_tokenizer.tokenize(text)

        print(f"Text: {text}")
        print(f"  Domain BPE ({len(domain_tokens)} tokens): {domain_tokens}")
        print(f"  MacBERT    ({len(bert_tokens)} tokens): {bert_tokens}")
        print()


if __name__ == "__main__":
    CORPUS_PATH = "data/douban_corpus.txt"
    TOKENIZER_OUTPUT = "models/douban_tokenizer"

    # Step 1: Export corpus
    doc_count = asyncio.run(export_corpus_from_db(CORPUS_PATH))
    print(f"Corpus: {doc_count} documents")

    if doc_count < 1000:
        print(
            f"Warning: Only {doc_count} documents in corpus. "
            "Train the tokenizer after accumulating 5000+ documents "
            "for meaningful domain vocabulary. Running anyway for demonstration."
        )

    # Step 2: Train tokenizer
    train_bpe_tokenizer(CORPUS_PATH, TOKENIZER_OUTPUT, vocab_size=8000)

    # Step 3: Analyze coverage on domain examples
    test_examples = [
        "流浪地球2的特效真的很震撼",
        "豆瓣评分被恶意刷低了很多",
        "yyds，这部剧真的太好看了",
        "楼主说的很有道理，支持一下",
    ]

    analyze_tokenizer_coverage(
        os.path.join(TOKENIZER_OUTPUT, "tokenizer.json"),
        test_examples
    )
```

### 15.3 Phase 13 Checklist

- [ ] Corpus exported from Postgres (at least 500 documents)
- [ ] BPE tokenizer trained without error
- [ ] Vocabulary size close to target (8000)
- [ ] Domain-specific multi-character tokens visible in vocabulary (e.g. `流浪`, `豆瓣`, `评分`)
- [ ] Coverage analysis shows domain tokenizer uses fewer tokens per text than character-level BERT

---

## 16. Phase 14 — Model Distillation

**Goal:** Compress your fine-tuned MacBERT sentiment classifier into a smaller, faster model using knowledge distillation. Understand how distillation works and measure the speed/accuracy tradeoff.

### 16.1 What Is Knowledge Distillation?

Fine-tuned MacBERT has 110M parameters and takes ~15ms per inference on CPU. For a 5-minute pipeline processing hundreds of threads per day, this is fine. But if you wanted to serve this model at web scale or on edge hardware, the latency and memory cost become prohibitive.

Knowledge distillation trains a small **student model** to mimic the behavior of a large **teacher model**. Instead of training on hard labels (0=negative, 1=positive), the student learns from the teacher's **soft probability outputs** — the full distribution like `[0.03, 0.97]` rather than just `[0, 1]`.

Why does this work better than training the student from scratch?
- The teacher's probability distribution contains more information than a hard label. A prediction of `[0.03, 0.97]` tells the student "this is strongly positive, and a little bit ambiguous" — information that the hard label `1` discards.
- The student learns to replicate the teacher's uncertainty, which transfers calibration and generalization.

### 16.2 Create `src/training/distill.py`

```python
"""
Distill fine-tuned MacBERT into a smaller BERT-tiny model.

Teacher: fine-tuned MacBERT-base (110M params, ~15ms/inference on CPU)
Student: bert-tiny (~4M params, ~2ms/inference on CPU)

Expected accuracy tradeoff: teacher ~93%, student ~88-90%
Speed gain: ~7-8x faster inference

Run on Kaggle T4 GPU. Estimated time: 30 minutes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments
from datasets import load_dataset
from loguru import logger

# ---- Configuration ----
TEACHER_MODEL = "./macbert_sentiment/best_model"  # Your fine-tuned model from Phase 8
STUDENT_MODEL = "huawei-noah/TinyBERT_General_4L_312D"  # 4-layer BERT-tiny
OUTPUT_DIR = "./distilled_sentiment"
TEMPERATURE = 4.0      # Softens probability distributions for richer signal.
                       # Higher T = softer targets = more information transferred.
                       # Lower T = harder targets. 4.0 is a standard starting value.
ALPHA = 0.7            # Weight for distillation loss vs. hard-label loss.
                       # 0.7 = 70% from teacher soft labels, 30% from true labels.
MAX_LENGTH = 256
BATCH_SIZE = 64        # Student is small, can use larger batches
LEARNING_RATE = 5e-5
NUM_EPOCHS = 5         # Students often need more epochs than teacher fine-tuning

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# ---- Load Teacher ----
print("Loading teacher model...")
teacher_tokenizer = AutoTokenizer.from_pretrained(TEACHER_MODEL)
teacher_model = AutoModelForSequenceClassification.from_pretrained(TEACHER_MODEL)
teacher_model.to(device)
teacher_model.eval()  # Teacher is frozen — only used to generate soft labels

# Freeze all teacher weights
for param in teacher_model.parameters():
    param.requires_grad = False

print(f"Teacher parameters: {sum(p.numel() for p in teacher_model.parameters()):,}")

# ---- Load Student ----
print("Loading student model...")
student_tokenizer = AutoTokenizer.from_pretrained(STUDENT_MODEL)
student_model = AutoModelForSequenceClassification.from_pretrained(
    STUDENT_MODEL,
    num_labels=2,
    ignore_mismatched_sizes=True  # Student has different architecture than teacher
)
student_model.to(device)

trainable = sum(p.numel() for p in student_model.parameters() if p.requires_grad)
print(f"Student trainable parameters: {trainable:,}")
print(f"Compression ratio: {sum(p.numel() for p in teacher_model.parameters()) / trainable:.1f}x")

# ---- Dataset ----
dataset = load_dataset("seamew/ChnSentiCorp")

def tokenize_for_student(examples):
    return student_tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding=False
    )

tokenized = dataset.map(tokenize_for_student, batched=True, remove_columns=["text"])
tokenized = tokenized.rename_column("label", "labels")
tokenized.set_format("torch")

# ---- Distillation Loss ----
def distillation_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    true_labels: torch.Tensor,
    temperature: float,
    alpha: float
) -> torch.Tensor:
    """
    Combined distillation + classification loss.

    The total loss has two components:
    1. Distillation loss: KL divergence between student and teacher soft probabilities.
       Computed at temperature T (both distributions are divided by T before softmax).
    2. Hard-label loss: standard cross-entropy against true labels.

    alpha controls the balance. Higher alpha = more learning from teacher.

    Args:
        student_logits: Raw outputs from student model
        teacher_logits: Raw outputs from teacher model (no gradient)
        true_labels: Ground truth integer labels
        temperature: Softening temperature T
        alpha: Weight of distillation loss

    Returns:
        Scalar loss tensor
    """
    # Soft targets from teacher (at temperature T)
    soft_teacher = F.softmax(teacher_logits / temperature, dim=-1)

    # Student soft predictions (at temperature T)
    soft_student = F.log_softmax(student_logits / temperature, dim=-1)

    # KL divergence loss — measures how different student distribution is from teacher
    # Multiply by T^2 to rescale gradients (standard in distillation literature)
    distill_loss = F.kl_div(
        soft_student,
        soft_teacher,
        reduction="batchmean"
    ) * (temperature ** 2)

    # Hard-label loss — standard cross-entropy
    hard_loss = F.cross_entropy(student_logits, true_labels)

    # Combined
    return alpha * distill_loss + (1 - alpha) * hard_loss


# ---- Custom Trainer ----
from transformers import Trainer
from torch.utils.data import DataLoader

class DistillationTrainer(Trainer):
    """
    Extends HuggingFace Trainer to use distillation loss instead of standard CE loss.
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")

        # Student forward pass
        student_outputs = model(**inputs)
        student_logits = student_outputs.logits

        # Teacher forward pass (no gradient — teacher is frozen)
        with torch.no_grad():
            # Re-tokenize for teacher (may have different vocab)
            # In practice, if teacher and student use same tokenizer, skip this.
            teacher_outputs = teacher_model(**inputs)
            teacher_logits = teacher_outputs.logits

        loss = distillation_loss(
            student_logits=student_logits,
            teacher_logits=teacher_logits,
            true_labels=labels,
            temperature=TEMPERATURE,
            alpha=ALPHA
        )

        return (loss, student_outputs) if return_outputs else loss


# ---- Training ----
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    evaluation_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    fp16=torch.cuda.is_available(),
    logging_steps=100,
    report_to="none"
)

def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted")
    }

from transformers import DataCollatorWithPadding
trainer = DistillationTrainer(
    model=student_model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=student_tokenizer,
    data_collator=DataCollatorWithPadding(student_tokenizer),
    compute_metrics=compute_metrics,
)

print("\nStarting distillation training...")
trainer.train()

# ---- Benchmark ----
print("\n=== Speed Benchmark ===")
import time

test_texts = ["这部电影太好看了，强烈推荐"] * 100

# Teacher speed
teacher_pipe = __import__("transformers").pipeline(
    "text-classification", model=teacher_model, tokenizer=teacher_tokenizer,
    device=-1, batch_size=32
)
start = time.time()
_ = teacher_pipe(test_texts)
teacher_time = time.time() - start

# Student speed
student_model_cpu = student_model.cpu()
from transformers import pipeline as hf_pipeline
student_pipe = hf_pipeline(
    "text-classification", model=student_model_cpu, tokenizer=student_tokenizer,
    device=-1, batch_size=32
)
start = time.time()
_ = student_pipe(test_texts)
student_time = time.time() - start

print(f"Teacher (MacBERT-base): {teacher_time:.2f}s for 100 texts ({teacher_time*10:.1f}ms each)")
print(f"Student (TinyBERT):     {student_time:.2f}s for 100 texts ({student_time*10:.1f}ms each)")
print(f"Speedup: {teacher_time/student_time:.1f}x")

trainer.save_model(f"{OUTPUT_DIR}/final")
student_tokenizer.save_pretrained(f"{OUTPUT_DIR}/final")
print(f"\nDistilled model saved to {OUTPUT_DIR}/final")
```

### 16.3 Evaluate the Tradeoff

After training, you should see roughly:

| Model | Accuracy | F1 | Inference (100 texts, CPU) |
|---|---|---|---|
| MacBERT-base (teacher) | ~93% | ~0.93 | ~12s |
| TinyBERT (student) | ~89–91% | ~0.90 | ~1.5s |

A 3–4% accuracy reduction for an 8x speed increase. For a 5-minute pipeline on a development machine, the teacher is perfectly adequate. The distilled student would matter if you wanted to serve the model in an API endpoint handling real-time traffic.

### 16.4 Phase 14 Checklist

- [ ] `distill.py` runs on Kaggle T4 without error
- [ ] `model.print_trainable_parameters()` shows student has far fewer params than teacher
- [ ] Distillation loss decreases over 5 epochs
- [ ] Student accuracy within 5% of teacher on test set
- [ ] Speedup benchmark confirms 6–10x faster inference
- [ ] Distilled model saved and downloadable

---

## 17. Reference: Key Concepts Glossary

**APScheduler:** A Python scheduling library with an asyncio backend. `AsyncIOScheduler` integrates into Python's async event loop, allowing scheduled tasks to run without blocking other async code. `max_instances=1` prevents a task from running concurrently with itself.

**BERTopic:** A neural topic modeling approach that uses sentence embeddings + UMAP dimensionality reduction + HDBSCAN clustering + c-TF-IDF topic representation. Produces semantically coherent topics without specifying the number of topics in advance.

**BPE (Byte-Pair Encoding):** A tokenization algorithm that starts with characters and iteratively merges the most frequent character pairs. The result is a vocabulary where common words are single tokens and rare words are broken into subword pieces.

**c-TF-IDF (Class-based TF-IDF):** BERTopic's topic representation method. Treats all documents in a cluster as one large "document" and computes TF-IDF relative to other clusters. Words that are common within a topic but rare across topics get high scores.

**HDBSCAN:** Hierarchical Density-Based Spatial Clustering of Applications with Noise. A clustering algorithm that finds clusters of arbitrary shape and marks low-density points as noise (label -1). Unlike k-means, it does not require specifying the number of clusters.

**Hypertable:** A TimescaleDB abstraction over a regular Postgres table. Data is automatically partitioned by time into chunks. Queries with time range filters only scan relevant chunks, not the entire table.

**jieba:** The most widely used Chinese word segmentation library. Uses an HMM (Hidden Markov Model) for unknown word discovery and a trie-based dictionary for known words. Works well on social media text after domain-specific vocabulary is added.

**Knowledge Distillation:** Training a small student model to mimic the output distribution of a large teacher model. The student learns from soft probability outputs (e.g., `[0.03, 0.97]`) rather than hard labels (`[0, 1]`), which carry more information.

**LoRA (Low-Rank Adaptation):** A parameter-efficient fine-tuning method that injects small trainable rank-decomposition matrices into a frozen pretrained model. Trains ~0.5% of parameters instead of 100%, making it feasible to fine-tune large models on free GPU tiers.

**MacBERT:** A Chinese BERT variant pretrained with MLM-as-Correction — instead of masking with `[MASK]`, it replaces tokens with semantically similar words. This makes the model more robust to natural language variation during fine-tuning.

**OpenCC:** Open Chinese Convert — a library for converting between Simplified and Traditional Chinese. Essential for normalizing user-generated content from mixed regional sources.

**Probing Classifier:** A simple classifier (logistic regression) trained on the internal representations of a neural network layer to measure what information is encoded there. If a probe achieves high accuracy, the layer encodes that property.

**Qdrant:** A purpose-built vector database using HNSW indexing for approximate nearest-neighbor search. Stores dense embedding vectors alongside metadata, enabling semantic similarity queries.

**TextRank:** A graph-based algorithm for keyword extraction and text summarization. Builds a word co-occurrence graph and applies PageRank to find the most central (important) words.

**TF-IDF (Term Frequency-Inverse Document Frequency):** A classic text representation. TF measures how often a word appears in a document; IDF penalizes words that appear in many documents. Words with high TF-IDF are distinctive to a specific document.

**TimescaleDB:** A Postgres extension adding time-series capabilities: hypertables with automatic time partitioning, continuous aggregates (pre-computed rollups), and time-aware query optimizations like `time_bucket()`.

**UMAP (Uniform Manifold Approximation and Projection):** A dimensionality reduction algorithm that preserves both local and global structure. Used in BERTopic to compress 384-dimensional embeddings to 5 dimensions for clustering.

**Whole-Word Masking (WWM):** A BERT pretraining strategy where entire words are masked rather than individual characters. The RoBERTa-wwm-ext and MacBERT models use this to learn better word-level representations for Chinese.

---

## 18. Reference: Common Errors & Fixes

### `asyncpg.exceptions.TooManyConnectionsError`

**Cause:** Too many concurrent database connections opened from async code.
**Fix:** Ensure you are using the shared `AsyncSessionLocal` session factory from `connection.py` everywhere. Never create a new engine per request.

```python
# Wrong: creates a new engine on every call
engine = create_async_engine(DATABASE_URL)

# Right: import the shared session factory
from db.connection import AsyncSessionLocal
async with AsyncSessionLocal() as session:
    ...
```

### `TokenizerWarning: Token indices sequence length is longer than the specified maximum`

**Cause:** A thread body is longer than the model's max_length (512 tokens for BERT).
**Fix:** Truncation is already enabled in the pipeline. If you see this in a notebook, add `truncation=True` to your tokenizer call.

### jieba Segmenting Single Characters Instead of Words

**Cause:** jieba's dictionary is not loaded for your domain terms. This commonly happens with proper nouns (movie titles, show names) not in jieba's default vocabulary.
**Fix:**
```python
import jieba
jieba.add_word("流浪地球", freq=100, tag="n")
jieba.add_word("豆瓣评分", freq=100, tag="n")
```
Build this list from your probing analysis — words that are single-character-split will show up as noise tokens.

### BERTopic Produces Only One Topic or All Outliers (-1)

**Cause 1:** Not enough data. BERTopic needs at least 50+ documents; 300+ for meaningful results.
**Cause 2:** `min_topic_size` is too large relative to your corpus.
**Fix:**
```python
# Reduce min_topic_size proportionally to corpus size
# Rule of thumb: min_topic_size ≈ corpus_size * 0.03
modeler = ChineseTopicModeler(min_topic_size=max(5, len(texts) // 30))
```

### Docker Container Exits Immediately

**Cause:** Usually a configuration error. Read the logs:
```bash
docker-compose logs timescaledb
docker-compose logs qdrant
```

**Common fix:** Environment variable not set in `.env`. Verify:
```bash
cat .env
docker-compose config  # Shows the resolved config with env vars substituted
```

### Streamlit `ProgrammingError: relation "keyword_timeseries" does not exist`

**Cause:** Schema migration did not run, usually because the Postgres container was started before the migrations volume mount was configured.
**Fix:**
```bash
# Apply migration manually
docker exec -i observatory_db psql -U observatory -d douban_observatory \
  < src/db/migrations/001_initial.sql
```

### Sentiment Model Returns All "neutral"

**Cause:** The confidence threshold in `analyze()` is set to 0.6 — if all predictions are below 0.6 confidence, they get downgraded to neutral.
**Fix:** This is usually a sign the model is not working correctly. Check:
1. The label normalization logic in `analyze()` — print raw `raw_results` to see what the model is actually returning
2. That you are feeding normalized Chinese text, not raw HTML or empty strings
3. If using the fine-tuned model, that the checkpoint loaded correctly

### APScheduler Cycle Overlap Warning Appearing Every Cycle

**Cause:** Your NLP pipeline is taking longer than 5 minutes per cycle.
**Diagnosis:** Check the cycle elapsed time in the logs:
```
Cycle 3 complete in 347.2s    ← 5.8 minutes: overlap risk
```
**Fix options:**
1. Skip summarization on CPU: the summarizer check in `_run_nlp` handles this automatically
2. Reduce `max_pages` from 3 to 1 (fewer threads per cycle)
3. Increase `SCRAPE_INTERVAL` in `.env` to 600 (10 minutes) if 5 minutes is too tight on your hardware

### Wordcloud Shows Boxes Instead of Chinese Characters

**Cause:** The wordcloud library does not have a Chinese font available.
**Fix:** Download and specify the font path:
```bash
curl -L "https://github.com/StellarCN/scp_zh/raw/master/fonts/simhei.ttf" -o data/simhei.ttf
```
Then in `app.py`:
```python
wc = WordCloud(font_path="data/simhei.ttf", ...)
```

### `OSError: [Errno 28] No space left on device` During Model Download

**Cause:** Transformer model files are large. MacBERT-base is ~400MB; Qwen2-1.5B is ~3GB.
**Fix:** Free disk space, or redirect the HuggingFace cache:
```bash
export TRANSFORMERS_CACHE=/path/to/large/disk/hf_cache
```
Or in Python before any import:
```python
import os
os.environ["TRANSFORMERS_CACHE"] = "/path/to/large/disk/hf_cache"
```

---

*End of Process Document. The pipeline you have built is genuinely production-scale: async ingestion, time-series storage, fine-tuned models, probing experiments, and a live dashboard. Everything from here is either expanding scope (more groups, more NLP tasks) or deepening the modeling work — both of which you now have the foundation for.*
