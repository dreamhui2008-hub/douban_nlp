import httpx
import asyncio
import os
from loguru import logger # Record events that happen while your program runs
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
        
        async def initalize(self):
                """Create the httpx client and attempt login."""
                self.client = httpx.AsyncClient( # Creates an HTTP async client object
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
                       data=login_data
                )

                # Verify login by checking for a user-specific cookie or redirect
                if "dbcl2" in self.client.cookies: # 'cookies' canbe used onto the self.client as a method because its part of the AsyncClient attribute
                       self.authenticated = True
                       logger.info("Douban login successful.")
                else:
                       logger.warning(
                       "Login may have failed — 'dbcl2' cookie not found. "
                       "Inspect the login response HTML for an error message."
                       )
            
            except Exception as e:
                   logger.error(f"Login failed: {e}")
        
        async def get(self, url: str, **kwarg) -> httpx.Response:
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



