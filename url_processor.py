from __future__ import annotations

import ipaddress
import json
import re
import socket
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup


class URLFetchError(Exception):
    """Raised when a job webpage cannot be fetched or parsed."""


# ============================================================
# CONFIGURATION
# ============================================================

MAX_TEXT_LENGTH = 150000
REQUEST_TIMEOUT = 20

USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/128.0.0.0 Safari/537.36"
)

HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": (
        "text/html,application/xhtml+xml,"
        "application/xml;q=0.9,image/avif,image/webp,"
        "*/*;q=0.8"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}


# ============================================================
# URL VALIDATION
# ============================================================

def _validate_url(url: str) -> str:
    """
    Validate and normalize a public HTTP/HTTPS URL.

    We do NOT reject a normal public domain simply because DNS
    returns an unexpected address. The actual HTTP request is
    allowed only for HTTP/HTTPS URLs and localhost is blocked.

    This avoids false 'private/restricted address' errors for
    legitimate sites such as Amazon, Microsoft and Naukri.
    """

    url = (url or "").strip()

    if not url:
        raise URLFetchError("Please enter a job URL.")

    # Allow users to enter:
    # www.amazon.jobs/...
    # instead of:
    # https://www.amazon.jobs/...
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "https://" + url

    parsed = urlparse(url)

    if parsed.scheme.lower() not in {"http", "https"}:
        raise URLFetchError(
            "Only HTTP and HTTPS job URLs are supported."
        )

    if not parsed.hostname:
        raise URLFetchError(
            "The supplied job URL is invalid."
        )

    hostname = parsed.hostname.lower().rstrip(".")

    # Block obvious local targets.
    blocked_hostnames = {
        "localhost",
        "localhost.localdomain",
        "ip6-localhost",
        "ip6-loopback",
    }

    if hostname in blocked_hostnames:
        raise URLFetchError(
            "Local/private URLs are not allowed."
        )

    # If the user directly supplies an IP address, reject
    # private/loopback/link-local/reserved addresses.
    try:
        direct_ip = ipaddress.ip_address(hostname)

        if (
            direct_ip.is_private
            or direct_ip.is_loopback
            or direct_ip.is_link_local
            or direct_ip.is_multicast
            or direct_ip.is_reserved
        ):
            raise URLFetchError(
                "The supplied URL points to a private or restricted address."
            )

    except ValueError:
        # Normal hostname. Continue.
        pass

    return url


# ============================================================
# TEXT CLEANING
# ============================================================

def _clean_text(text: str) -> str:
    """Normalize and clean extracted webpage text."""

    if not text:
        return ""

    text = str(text)

    # Normalize line endings.
    text = re.sub(r"\r\n?", "\n", text)

    # Remove excessive spaces.
    text = re.sub(r"[ \t]+", " ", text)

    # Remove excessive blank lines.
    text = re.sub(r"\n\s*\n+", "\n\n", text)

    cleaned_lines = []

    for line in text.splitlines():
        line = line.strip()

        if not line:
            continue

        # Ignore extremely long garbage lines.
        if len(line) > 20000:
            line = line[:20000]

        cleaned_lines.append(line)

    text = "\n".join(cleaned_lines)

    return text[:MAX_TEXT_LENGTH].strip()


# ============================================================
# JSON-LD EXTRACTION
# ============================================================

def _extract_json_ld(soup: BeautifulSoup) -> tuple[str, str]:
    """
    Extract JobPosting information from JSON-LD.

    Many professional job websites expose structured data like:

    {
        "@type": "JobPosting",
        "title": "...",
        "description": "...",
        "hiringOrganization": {...},
        "jobLocation": {...}
    }
    """

    pieces = []
    title = ""

    scripts = soup.find_all(
        "script",
        attrs={"type": re.compile(r"application/ld\+json", re.I)}
    )

    for script in scripts:
        raw = script.string or script.get_text()

        if not raw:
            continue

        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            continue

        objects = []

        if isinstance(data, dict):
            objects.append(data)

            graph = data.get("@graph")

            if isinstance(graph, list):
                objects.extend(graph)

        elif isinstance(data, list):
            objects.extend(data)

        for item in objects:
            if not isinstance(item, dict):
                continue

            item_type = item.get("@type")

            if isinstance(item_type, list):
                is_job = "JobPosting" in item_type
            else:
                is_job = item_type == "JobPosting"

            if not is_job:
                continue

            # ------------------------------------------------
            # Job title
            # ------------------------------------------------
            job_title = item.get("title")

            if job_title and not title:
                title = str(job_title).strip()

            if job_title:
                pieces.append(
                    f"Job Title: {job_title}"
                )

            # ------------------------------------------------
            # Description
            # ------------------------------------------------
            description = item.get("description")

            if description:
                description_soup = BeautifulSoup(
                    str(description),
                    "html.parser"
                )

                description_text = description_soup.get_text(
                    "\n",
                    strip=True
                )

                if description_text:
                    pieces.append(
                        f"Job Description:\n{description_text}"
                    )

            # ------------------------------------------------
            # Organization
            # ------------------------------------------------
            organization = item.get("hiringOrganization")

            if isinstance(organization, dict):
                organization_name = organization.get("name")

                if organization_name:
                    pieces.append(
                        f"Company: {organization_name}"
                    )

            elif organization:
                pieces.append(
                    f"Company: {organization}"
                )

            # ------------------------------------------------
            # Location
            # ------------------------------------------------
            location = item.get("jobLocation")

            if location:
                location_text = _json_location_to_text(location)

                if location_text:
                    pieces.append(
                        f"Location: {location_text}"
                    )

            # ------------------------------------------------
            # Employment type
            # ------------------------------------------------
            employment_type = item.get("employmentType")

            if employment_type:
                pieces.append(
                    f"Employment Type: {employment_type}"
                )

            # ------------------------------------------------
            # Salary
            # ------------------------------------------------
            salary = item.get("baseSalary")

            if salary:
                salary_text = _json_salary_to_text(salary)

                if salary_text:
                    pieces.append(
                        f"Salary: {salary_text}"
                    )

    return _clean_text("\n".join(pieces)), title


def _json_location_to_text(location) -> str:
    """Convert JSON-LD job location data to readable text."""

    if isinstance(location, list):
        values = [
            _json_location_to_text(item)
            for item in location
        ]

        return ", ".join(
            value for value in values if value
        )

    if not isinstance(location, dict):
        return str(location)

    address = location.get("address")

    if isinstance(address, dict):
        parts = []

        for key in [
            "streetAddress",
            "addressLocality",
            "addressRegion",
            "postalCode",
            "addressCountry",
        ]:
            value = address.get(key)

            if value:
                parts.append(str(value))

        return ", ".join(parts)

    if address:
        return str(address)

    return ""


def _json_salary_to_text(salary) -> str:
    """Convert JSON-LD salary data to readable text."""

    if not isinstance(salary, dict):
        return str(salary)

    currency = salary.get("currency", "")

    value = salary.get("value")

    if isinstance(value, dict):
        minimum = value.get("minValue")
        maximum = value.get("maxValue")

        if minimum is not None and maximum is not None:
            return f"{minimum}-{maximum} {currency}".strip()

        if minimum is not None:
            return f"{minimum} {currency}".strip()

        if maximum is not None:
            return f"{maximum} {currency}".strip()

    if value is not None:
        return f"{value} {currency}".strip()

    return ""


# ============================================================
# HTML TEXT EXTRACTION
# ============================================================

def _extract_text_from_html(html: str) -> tuple[str, str]:
    """
    Extract useful readable content from an HTML document.

    Works with:
    - Amazon
    - Microsoft
    - IBM
    - Naukri
    - generic job sites
    - JSON-LD JobPosting pages
    """

    if not html:
        return "", ""

    soup = BeautifulSoup(
        html,
        "html.parser"
    )

    # --------------------------------------------------------
    # Page title
    # --------------------------------------------------------

    title = ""

    if soup.title:
        title = soup.title.get_text(
            " ",
            strip=True
        )

    # --------------------------------------------------------
    # First collect JSON-LD before removing scripts.
    # --------------------------------------------------------

    json_ld_text, json_ld_title = _extract_json_ld(
        soup
    )

    if json_ld_title:
        title = json_ld_title

    # --------------------------------------------------------
    # Remove useless elements.
    # --------------------------------------------------------

    for tag in soup(
        [
            "script",
            "style",
            "noscript",
            "svg",
            "canvas",
            "iframe",
            "template",
        ]
    ):
        tag.decompose()

    # --------------------------------------------------------
    # Remove common navigation/footer areas.
    # --------------------------------------------------------

    for selector in [
        "nav",
        "footer",
        "header",
        ".navbar",
        ".navigation",
        ".nav",
        ".footer",
        ".cookie",
        ".cookie-banner",
        ".cookie-consent",
        ".advertisement",
        ".ads",
        ".advert",
        ".social-share",
    ]:
        try:
            for element in soup.select(selector):
                element.decompose()
        except Exception:
            pass

    # --------------------------------------------------------
    # Job-specific selectors.
    #
    # Different sites use different class names.
    # --------------------------------------------------------

    selectors = [
        # Generic
        "main",
        "article",
        "[role='main']",

        # Generic job descriptions
        ".job-description",
        "#job-description",
        ".jobDescription",
        ".job-description-container",
        ".job-detail",
        ".job-details",
        ".job-detail-content",
        ".description",
        "#description",

        # Naukri
        ".styles_JDC__dang-inner-html__h0K4t",
        ".dang-inner-html",
        ".jd-container",
        ".job-desc",
        ".job-desc-container",
        ".job-description-content",
        ".jd-desc",
        ".jd-section",
        ".jobDescriptionContent",

        # Amazon
        "#job-detail-apply",
        "#job-detail",
        ".job-detail",
        ".job-detail-apply",

        # Indeed
        "#jobDescriptionText",
        ".jobsearch-jobDescriptionText",

        # LinkedIn-like structures
        ".description__text",
        ".show-more-less-html__markup",

        # Other common names
        ".job-content",
        ".job-content-container",
        ".job-posting",
        ".job-posting-content",
        ".posting-content",
        ".job-info",
        ".job-information",
        ".career-detail",
        ".career-details",
    ]

    candidates = []

    for selector in selectors:

        try:
            elements = soup.select(selector)

        except Exception:
            continue

        for element in elements:

            text = element.get_text(
                "\n",
                strip=True
            )

            text = _clean_text(text)

            if len(text) >= 150:
                candidates.append(text)

    # --------------------------------------------------------
    # Full visible body fallback.
    # --------------------------------------------------------

    body_text = ""

    if soup.body:
        body_text = soup.body.get_text(
            "\n",
            strip=True
        )

        body_text = _clean_text(
            body_text
        )

    # --------------------------------------------------------
    # Choose useful candidate.
    #
    # We don't blindly choose the largest page because some
    # career pages contain huge navigation text.
    # --------------------------------------------------------

    selected_text = ""

    if candidates:

        # Score candidates by job-related terms + length.
        def candidate_score(text):

            lowered = text.lower()

            job_words = [
                "job description",
                "responsibilities",
                "qualifications",
                "requirements",
                "skills",
                "experience",
                "education",
                "location",
                "salary",
                "employment",
                "about the job",
                "what you'll do",
                "what you will do",
                "apply",
                "software engineer",
                "developer",
                "analyst",
                "consultant",
                "manager",
                "intern",
            ]

            matches = sum(
                1
                for word in job_words
                if word in lowered
            )

            # Give useful text a score based on both
            # job terminology and length.
            return (
                matches * 500
                + min(len(text), 30000)
            )

        selected_text = max(
            candidates,
            key=candidate_score
        )

    # --------------------------------------------------------
    # Combine JSON-LD + HTML content.
    # --------------------------------------------------------

    parts = []

    if json_ld_text:
        parts.append(json_ld_text)

    if selected_text:
        parts.append(selected_text)

    # If no special job container was found, use body.
    if not selected_text and body_text:
        parts.append(body_text)

    combined = _clean_text(
        "\n\n".join(parts)
    )

    return combined, title


# ============================================================
# JOB PAGE HEURISTIC
# ============================================================

def _looks_like_job_page(text: str) -> bool:
    """
    Determine whether extracted content contains enough
    job-related information.

    This is intentionally broad so that different job portals
    are accepted.
    """

    if not text:
        return False

    # Very short pages are unlikely to contain a job description.
    if len(text) < 350:
        return False

    lowered = text.lower()

    job_terms = [
        "job description",
        "responsibilities",
        "requirements",
        "qualifications",
        "skills",
        "experience",
        "education",
        "location",
        "salary",
        "employment",
        "employment type",
        "about the job",
        "what you'll do",
        "what you will do",
        "role",
        "career",
        "apply",
        "job type",
        "full time",
        "full-time",
        "part time",
        "part-time",
        "software engineer",
        "software developer",
        "developer",
        "engineer",
        "analyst",
        "consultant",
        "manager",
        "intern",
        "job title",
        "company",
    ]

    matches = sum(
        1
        for term in job_terms
        if term in lowered
    )

    # Normal job pages should contain at least a few
    # recognizable job terms.
    return matches >= 2


# ============================================================
# NORMAL HTTP FETCH
# ============================================================

def _fetch_with_requests(
    url: str
) -> tuple[str, str, str]:
    """
    Fetch a normal server-rendered webpage.
    """

    try:

        response = requests.get(
            url,
            headers=HEADERS,
            timeout=REQUEST_TIMEOUT,
            allow_redirects=True,
        )

    except requests.RequestException as exc:

        raise URLFetchError(
            f"Unable to fetch the job webpage: {exc}"
        ) from exc

    # --------------------------------------------------------
    # HTTP status
    # --------------------------------------------------------

    if response.status_code >= 400:

        raise URLFetchError(
            f"The job webpage returned HTTP "
            f"{response.status_code}."
        )

    # --------------------------------------------------------
    # Content type
    # --------------------------------------------------------

    content_type = (
        response.headers
        .get("Content-Type", "")
        .lower()
    )

    if (
        "text/html" not in content_type
        and "application/xhtml" not in content_type
    ):

        raise URLFetchError(
            "The supplied URL does not appear "
            "to contain an HTML webpage."
        )

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    text, title = _extract_text_from_html(
        response.text
    )

    return (
        text,
        title,
        response.url,
    )


# ============================================================
# JAVASCRIPT / PLAYWRIGHT FETCH
# ============================================================

def _fetch_with_playwright(
    url: str
) -> tuple[str, str, str]:
    """
    Render JavaScript-heavy job pages using Chromium.

    Uses both:
        1. rendered HTML extraction
        2. direct visible-text extraction

    The second method is useful for portals such as Naukri
    where the final DOM may not match predictable CSS
    selectors.
    """

    try:
        from playwright.sync_api import (
            TimeoutError as PlaywrightTimeoutError,
            sync_playwright,
        )

    except ImportError as exc:
        raise URLFetchError(
            "This job page requires JavaScript rendering. "
            "Install it with: "
            "pip install playwright && "
            "python3 -m playwright install chromium"
        ) from exc

    try:

        with sync_playwright() as playwright:

            browser = playwright.chromium.launch(
                headless=True
            )

            context = browser.new_context(
                user_agent=USER_AGENT,
                viewport={
                    "width": 1440,
                    "height": 1000,
                },
                locale="en-US",
                java_script_enabled=True,
            )

            page = context.new_page()

            # ------------------------------------------------
            # Load page
            # ------------------------------------------------

            page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=30000,
            )

            # ------------------------------------------------
            # Allow JavaScript to finish.
            # ------------------------------------------------

            try:
                page.wait_for_load_state(
                    "networkidle",
                    timeout=12000,
                )
            except PlaywrightTimeoutError:
                pass

            # Extra time for portals that load job data
            # through delayed API calls.
            page.wait_for_timeout(5000)

            # ------------------------------------------------
            # Scroll through the page.
            # ------------------------------------------------

            try:
                page.evaluate(
                    """
                    async () => {
                        await new Promise((resolve) => {
                            let total = 0;
                            const distance = 400;

                            const timer = setInterval(() => {
                                window.scrollBy(
                                    0,
                                    distance
                                );

                                total += distance;

                                if (
                                    total >=
                                    document.body.scrollHeight
                                ) {
                                    clearInterval(timer);
                                    resolve();
                                }
                            }, 150);
                        });
                    }
                    """
                )

                page.wait_for_timeout(2000)

            except Exception:
                pass

            # ------------------------------------------------
            # Get rendered HTML.
            # ------------------------------------------------

            html = page.content()

            final_url = page.url

            try:
                title = page.title()
            except Exception:
                title = ""

            # ------------------------------------------------
            # IMPORTANT:
            # Get the actual visible text from Chromium.
            #
            # This can recover Naukri content even when its
            # CSS class names don't match our selectors.
            # ------------------------------------------------

            visible_text = ""

            try:
                visible_text = page.locator("body").inner_text(
                    timeout=5000
                )
            except Exception:
                try:
                    visible_text = page.evaluate(
                        """
                        () => document.body
                            ? document.body.innerText
                            : ""
                        """
                    )
                except Exception:
                    visible_text = ""

            context.close()
            browser.close()

    except PlaywrightTimeoutError as exc:

        raise URLFetchError(
            "The job webpage took too long to load."
        ) from exc

    except Exception as exc:

        raise URLFetchError(
            f"Unable to render the job webpage: {exc}"
        ) from exc

    # ========================================================
    # EXTRACTION METHOD 1
    # Rendered HTML
    # ========================================================

    html_text, extracted_title = _extract_text_from_html(
        html
    )

    if extracted_title:
        title = extracted_title

    # ========================================================
    # EXTRACTION METHOD 2
    # Direct browser visible text
    # ========================================================

    visible_text = _clean_text(
        visible_text
    )

    # --------------------------------------------------------
    # Prefer the extraction that actually looks like a job.
    # --------------------------------------------------------

    candidates = [
        html_text,
        visible_text,
    ]

    valid_candidates = [
        candidate
        for candidate in candidates
        if _looks_like_job_page(candidate)
    ]

    if valid_candidates:

        # Prefer the richest useful candidate.
        text = max(
            valid_candidates,
            key=len
        )

    else:

        # Return the richer candidate so the caller can
        # provide the normal extraction error.
        text = max(
            candidates,
            key=len,
            default=""
        )

    return (
        text,
        title,
        final_url,
    )
    """
    Render JavaScript-heavy job pages using Chromium.

    This is especially useful for:
    - Naukri
    - dynamically rendered career pages
    - pages where requests.get() receives only a shell
    """

    try:

        from playwright.sync_api import (
            TimeoutError as PlaywrightTimeoutError,
            sync_playwright,
        )

    except ImportError as exc:

        raise URLFetchError(
            "This job page requires JavaScript rendering. "
            "Install it with: "
            "pip install playwright && "
            "python3 -m playwright install chromium"
        ) from exc

    browser = None

    try:

        with sync_playwright() as playwright:

            browser = playwright.chromium.launch(
                headless=True
            )

            context = browser.new_context(
                user_agent=USER_AGENT,
                viewport={
                    "width": 1440,
                    "height": 1000,
                },
                locale="en-US",
            )

            page = context.new_page()

            # ------------------------------------------------
            # Load page
            # ------------------------------------------------

            page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=30000,
            )

            # ------------------------------------------------
            # Wait for dynamic content.
            # ------------------------------------------------

            try:

                page.wait_for_load_state(
                    "networkidle",
                    timeout=12000,
                )

            except PlaywrightTimeoutError:
                # Some job sites keep network connections alive.
                # This does not necessarily mean the page failed.
                pass

            # ------------------------------------------------
            # Extra rendering time.
            # ------------------------------------------------

            page.wait_for_timeout(
                3000
            )

            # ------------------------------------------------
            # Scroll to trigger lazy-loaded content.
            # ------------------------------------------------

            try:

                page.evaluate(
                    """
                    async () => {
                        await new Promise((resolve) => {
                            let total = 0;
                            const distance = 500;

                            const timer = setInterval(() => {
                                window.scrollBy(
                                    0,
                                    distance
                                );

                                total += distance;

                                if (
                                    total >=
                                    document.body.scrollHeight
                                ) {
                                    clearInterval(timer);
                                    resolve();
                                }
                            }, 150);
                        });
                    }
                    """
                )

                page.wait_for_timeout(
                    1500
                )

            except Exception:
                pass

            # ------------------------------------------------
            # Get rendered HTML.
            # ------------------------------------------------

            html = page.content()

            final_url = page.url

            try:
                title = page.title()
            except Exception:
                title = ""

            context.close()
            browser.close()
            browser = None

    except PlaywrightTimeoutError as exc:

        if browser:
            try:
                browser.close()
            except Exception:
                pass

        raise URLFetchError(
            "The job webpage took too long to load."
        ) from exc

    except Exception as exc:

        if browser:
            try:
                browser.close()
            except Exception:
                pass

        raise URLFetchError(
            f"Unable to render the job webpage: {exc}"
        ) from exc

    # --------------------------------------------------------
    # Extract rendered content.
    # --------------------------------------------------------

    text, extracted_title = _extract_text_from_html(
        html
    )

    if extracted_title:
        title = extracted_title

    return (
        text,
        title,
        final_url,
    )


# ============================================================
# MAIN FUNCTION
# ============================================================

def fetch_job_page(url: str) -> dict:
    """
    Fetch a public job webpage and extract readable job text.

    Strategy:

        1. Validate URL.
        2. Try normal HTTP request.
        3. Check extracted content.
        4. If insufficient, use Playwright.
        5. Extract JSON-LD + HTML content.
        6. Return text and metadata.

    This function is used by app.py and does not change
    job_analyzer.py or the ML detector.
    """

    url = _validate_url(url)

    request_error = None

    # ========================================================
    # STEP 1 — NORMAL HTTP REQUEST
    # ========================================================

    try:

        text, title, final_url = _fetch_with_requests(
            url
        )

        if _looks_like_job_page(text):

            parsed = urlparse(
                final_url
            )

            return {
                "url": final_url,
                "title": title[:255],
                "text": text,
                "domain": parsed.netloc,
            }

    except URLFetchError as exc:

        request_error = exc

    # ========================================================
    # STEP 2 — JAVASCRIPT / BROWSER FALLBACK
    # ========================================================

    try:

        text, title, final_url = _fetch_with_playwright(
            url
        )

        if not _looks_like_job_page(text):

            raise URLFetchError(
                "The webpage did not contain enough "
                "readable job information even after "
                "JavaScript rendering."
            )

        parsed = urlparse(
            final_url
        )

        return {
            "url": final_url,
            "title": title[:255],
            "text": text,
            "domain": parsed.netloc,
        }

    except URLFetchError as browser_error:

        # ----------------------------------------------------
        # Give the user one clear message.
        # ----------------------------------------------------

        raise URLFetchError(
            "Unable to extract enough readable job "
            "information from this webpage. "
            "The site may be JavaScript-rendered, "
            "blocked, or may not contain a public job "
            "description."
        ) from browser_error