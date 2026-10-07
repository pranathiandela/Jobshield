from __future__ import annotations

import ipaddress
import json
import re
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
    """

    url = (url or "").strip()

    if not url:
        raise URLFetchError(
            "Please enter a job URL."
        )

    if not re.match(
        r"^https?://",
        url,
        re.IGNORECASE,
    ):
        url = "https://" + url

    parsed = urlparse(url)

    if parsed.scheme.lower() not in {
        "http",
        "https",
    }:
        raise URLFetchError(
            "Only HTTP and HTTPS job URLs are supported."
        )

    if not parsed.hostname:
        raise URLFetchError(
            "The supplied job URL is invalid."
        )

    hostname = (
        parsed.hostname
        .lower()
        .rstrip(".")
    )

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

    try:

        direct_ip = ipaddress.ip_address(
            hostname
        )

        if (
            direct_ip.is_private
            or direct_ip.is_loopback
            or direct_ip.is_link_local
            or direct_ip.is_multicast
            or direct_ip.is_reserved
        ):
            raise URLFetchError(
                "The supplied URL points to a "
                "private or restricted address."
            )

    except ValueError:
        pass

    return url


# ============================================================
# TEXT CLEANING
# ============================================================

def _clean_text(text: str) -> str:
    """
    Normalize and clean extracted webpage text.
    """

    if not text:
        return ""

    text = str(text)

    text = re.sub(
        r"\r\n?",
        "\n",
        text,
    )

    text = re.sub(
        r"[ \t]+",
        " ",
        text,
    )

    text = re.sub(
        r"\n\s*\n+",
        "\n\n",
        text,
    )

    cleaned_lines = []

    for line in text.splitlines():

        line = line.strip()

        if not line:
            continue

        if len(line) > 20000:
            line = line[:20000]

        cleaned_lines.append(line)

    text = "\n".join(
        cleaned_lines
    )

    return text[
        :MAX_TEXT_LENGTH
    ].strip()


# ============================================================
# JSON-LD EXTRACTION
# ============================================================

def _extract_json_ld(
    soup: BeautifulSoup,
) -> tuple[str, str]:
    """
    Extract JobPosting information from JSON-LD.
    """

    pieces = []
    title = ""

    scripts = soup.find_all(
        "script",
        attrs={
            "type": re.compile(
                r"application/ld\+json",
                re.I,
            )
        },
    )

    for script in scripts:

        raw = (
            script.string
            or script.get_text()
        )

        if not raw:
            continue

        try:

            data = json.loads(
                raw
            )

        except (
            json.JSONDecodeError,
            TypeError,
        ):
            continue

        objects = []

        if isinstance(
            data,
            dict,
        ):

            objects.append(data)

            graph = data.get(
                "@graph"
            )

            if isinstance(
                graph,
                list,
            ):
                objects.extend(graph)

        elif isinstance(
            data,
            list,
        ):

            objects.extend(data)

        for item in objects:

            if not isinstance(
                item,
                dict,
            ):
                continue

            item_type = item.get(
                "@type"
            )

            if isinstance(
                item_type,
                list,
            ):

                is_job = (
                    "JobPosting"
                    in item_type
                )

            else:

                is_job = (
                    item_type
                    == "JobPosting"
                )

            if not is_job:
                continue

            # ------------------------------------------------
            # TITLE
            # ------------------------------------------------

            job_title = item.get(
                "title"
            )

            if (
                job_title
                and not title
            ):

                title = str(
                    job_title
                ).strip()

            if job_title:

                pieces.append(
                    f"Job Title: {job_title}"
                )

            # ------------------------------------------------
            # DESCRIPTION
            # ------------------------------------------------

            description = item.get(
                "description"
            )

            if description:

                description_soup = (
                    BeautifulSoup(
                        str(description),
                        "html.parser",
                    )
                )

                description_text = (
                    description_soup.get_text(
                        "\n",
                        strip=True,
                    )
                )

                if description_text:

                    pieces.append(
                        "Job Description:\n"
                        f"{description_text}"
                    )

            # ------------------------------------------------
            # ORGANIZATION
            # ------------------------------------------------

            organization = item.get(
                "hiringOrganization"
            )

            if isinstance(
                organization,
                dict,
            ):

                organization_name = (
                    organization.get(
                        "name"
                    )
                )

                if organization_name:

                    pieces.append(
                        "Company: "
                        f"{organization_name}"
                    )

            elif organization:

                pieces.append(
                    "Company: "
                    f"{organization}"
                )

            # ------------------------------------------------
            # LOCATION
            # ------------------------------------------------

            location = item.get(
                "jobLocation"
            )

            if location:

                location_text = (
                    _json_location_to_text(
                        location
                    )
                )

                if location_text:

                    pieces.append(
                        "Location: "
                        f"{location_text}"
                    )

            # ------------------------------------------------
            # EMPLOYMENT TYPE
            # ------------------------------------------------

            employment_type = item.get(
                "employmentType"
            )

            if employment_type:

                if isinstance(
                    employment_type,
                    list,
                ):

                    employment_type = (
                        ", ".join(
                            map(
                                str,
                                employment_type,
                            )
                        )
                    )

                pieces.append(
                    "Employment Type: "
                    f"{employment_type}"
                )

            # ------------------------------------------------
            # SALARY
            # ------------------------------------------------

            salary = item.get(
                "baseSalary"
            )

            if salary:

                salary_text = (
                    _json_salary_to_text(
                        salary
                    )
                )

                if salary_text:

                    pieces.append(
                        "Salary: "
                        f"{salary_text}"
                    )

    return (
        _clean_text(
            "\n".join(pieces)
        ),
        title,
    )


# ============================================================
# JSON-LD LOCATION
# ============================================================

def _json_location_to_text(
    location,
) -> str:
    """
    Convert JSON-LD job location to readable text.
    """

    if isinstance(
        location,
        list,
    ):

        values = [
            _json_location_to_text(
                item
            )
            for item in location
        ]

        return ", ".join(
            value
            for value in values
            if value
        )

    if not isinstance(
        location,
        dict,
    ):

        return str(location)

    address = location.get(
        "address"
    )

    if isinstance(
        address,
        dict,
    ):

        parts = []

        for key in [
            "streetAddress",
            "addressLocality",
            "addressRegion",
            "postalCode",
            "addressCountry",
        ]:

            value = address.get(
                key
            )

            if value:
                parts.append(
                    str(value)
                )

        return ", ".join(parts)

    if address:
        return str(address)

    return ""


# ============================================================
# JSON-LD SALARY
# ============================================================

def _json_salary_to_text(
    salary,
) -> str:
    """
    Convert JSON-LD salary to readable text.
    """

    if not isinstance(
        salary,
        dict,
    ):
        return str(salary)

    currency = salary.get(
        "currency",
        "",
    )

    value = salary.get(
        "value"
    )

    if isinstance(
        value,
        dict,
    ):

        minimum = value.get(
            "minValue"
        )

        maximum = value.get(
            "maxValue"
        )

        if (
            minimum is not None
            and maximum is not None
        ):

            return (
                f"{minimum}-{maximum} "
                f"{currency}"
            ).strip()

        if minimum is not None:

            return (
                f"{minimum} "
                f"{currency}"
            ).strip()

        if maximum is not None:

            return (
                f"{maximum} "
                f"{currency}"
            ).strip()

    if value is not None:

        return (
            f"{value} "
            f"{currency}"
        ).strip()

    return ""


# ============================================================
# META / OPEN GRAPH EXTRACTION
# ============================================================

def _extract_meta_information(
    soup: BeautifulSoup,
) -> str:
    """
    Extract useful job-related information from meta tags.

    Some career sites put a meaningful job description or
    summary inside description / OG tags even when the main
    HTML structure is difficult to parse.
    """

    pieces = []

    meta_names = [
        "description",
        "og:description",
        "twitter:description",
    ]

    for name in meta_names:

        tag = soup.find(
            "meta",
            attrs={
                "name": name,
            },
        )

        if not tag:

            tag = soup.find(
                "meta",
                attrs={
                    "property": name,
                },
            )

        if not tag:
            continue

        content = tag.get(
            "content",
            "",
        )

        content = _clean_text(
            content
        )

        if (
            content
            and len(content) >= 50
        ):

            pieces.append(
                content
            )

    return _clean_text(
        "\n".join(pieces)
    )


# ============================================================
# JOB-RELATED TERMS
# ============================================================

JOB_TERMS = [
    "job description",
    "job overview",
    "job summary",
    "responsibilities",
    "requirements",
    "qualifications",
    "skills",
    "experience",
    "education",
    "location",
    "salary",
    "compensation",
    "employment",
    "employment type",
    "about the job",
    "about this job",
    "about the role",
    "role overview",
    "what you'll do",
    "what you will do",
    "what you'll be doing",
    "what you will be doing",
    "what you'll bring",
    "what you will bring",
    "who you are",
    "who we're looking for",
    "who we are looking for",
    "key responsibilities",
    "key requirements",
    "preferred qualifications",
    "minimum qualifications",
    "required qualifications",
    "required skills",
    "preferred skills",
    "preferred experience",
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
    "apply",
    "career",
    "careers",
    "candidate",
    "position",
    "vacancy",
    "opening",
]


# ============================================================
# JOB TERM COUNT
# ============================================================

def _job_term_count(
    text: str,
) -> int:
    """
    Count recognizable job-related terms.
    """

    if not text:
        return 0

    lowered = text.lower()

    return sum(
        1
        for term in JOB_TERMS
        if term in lowered
    )


# ============================================================
# JOB CONTENT SCORE
# ============================================================

def _job_content_score(
    text: str,
) -> int:
    """
    Give extracted text a usefulness score.

    This is intentionally separate from the final validation
    so that we can choose the best candidate among several
    extraction methods.
    """

    if not text:
        return 0

    length_score = min(
        len(text),
        30000,
    )

    term_score = (
        _job_term_count(text)
        * 700
    )

    return (
        term_score
        + length_score
    )


# ============================================================
# JOB PAGE HEURISTIC
# ============================================================

def _looks_like_job_page(
    text: str,
) -> bool:
    """
    Determine whether extracted content contains enough
    recognizable job information.
    """

    if not text:
        return False

    if len(text) < 350:
        return False

    lowered = text.lower()

    matches = _job_term_count(
        text
    )

    # --------------------------------------------------------
    # Strong job-description indicators.
    # --------------------------------------------------------

    strong_indicators = [
        "job description",
        "responsibilities",
        "qualifications",
        "requirements",
        "what you'll do",
        "what you will do",
        "key responsibilities",
        "preferred qualifications",
        "minimum qualifications",
        "required skills",
        "job overview",
        "about the role",
        "about the job",
    ]

    strong_matches = sum(
        1
        for term in strong_indicators
        if term in lowered
    )

    # --------------------------------------------------------
    # A rich page containing multiple job terms is accepted.
    # --------------------------------------------------------

    if matches >= 3:
        return True

    # --------------------------------------------------------
    # Two strong job indicators are enough.
    # --------------------------------------------------------

    if strong_matches >= 2:
        return True

    # --------------------------------------------------------
    # A long page with two job terms is acceptable.
    # --------------------------------------------------------

    if (
        len(text) >= 1200
        and matches >= 2
    ):
        return True

    return False


# ============================================================
# HTML TEXT EXTRACTION
# ============================================================

def _extract_text_from_html(
    html: str,
) -> tuple[str, str]:
    """
    Extract useful readable content from an HTML document.

    Supports:
    - JSON-LD JobPosting
    - Lever
    - Microsoft
    - Amazon
    - Naukri
    - Indeed
    - LinkedIn-like structures
    - generic career pages
    """

    if not html:
        return "", ""

    soup = BeautifulSoup(
        html,
        "html.parser",
    )

    # --------------------------------------------------------
    # PAGE TITLE
    # --------------------------------------------------------

    title = ""

    if soup.title:

        title = soup.title.get_text(
            " ",
            strip=True,
        )

    # --------------------------------------------------------
    # JSON-LD
    # --------------------------------------------------------

    json_ld_text, json_ld_title = (
        _extract_json_ld(
            soup
        )
    )

    if json_ld_title:
        title = json_ld_title

    # --------------------------------------------------------
    # META INFORMATION
    # --------------------------------------------------------

    meta_text = (
        _extract_meta_information(
            soup
        )
    )

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
    # Remove navigation/footer areas.
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

            for element in soup.select(
                selector
            ):

                element.decompose()

        except Exception:
            pass

    # --------------------------------------------------------
    # JOB-SPECIFIC SELECTORS
    # --------------------------------------------------------

    selectors = [

        # ----------------------------------------------------
        # Generic
        # ----------------------------------------------------

        "main",
        "article",
        "[role='main']",

        # ----------------------------------------------------
        # Generic job descriptions
        # ----------------------------------------------------

        ".job-description",
        "#job-description",
        ".jobDescription",
        ".job-description-container",
        ".job-detail",
        ".job-details",
        ".job-detail-content",
        ".description",
        "#description",

        # ----------------------------------------------------
        # Lever
        # ----------------------------------------------------

        ".posting-page",
        ".posting-headline",
        ".posting-description",
        ".posting-categories",
        ".posting-content",
        ".section-wrapper",
        ".content",

        # ----------------------------------------------------
        # Naukri
        # ----------------------------------------------------

        ".styles_JDC__dang-inner-html__h0K4t",
        ".dang-inner-html",
        ".jd-container",
        ".job-desc",
        ".job-desc-container",
        ".job-description-content",
        ".jd-desc",
        ".jd-section",
        ".jobDescriptionContent",

        # ----------------------------------------------------
        # Amazon
        # ----------------------------------------------------

        "#job-detail-apply",
        "#job-detail",
        ".job-detail",
        ".job-detail-apply",

        # ----------------------------------------------------
        # Indeed
        # ----------------------------------------------------

        "#jobDescriptionText",
        ".jobsearch-jobDescriptionText",

        # ----------------------------------------------------
        # LinkedIn-like
        # ----------------------------------------------------

        ".description__text",
        ".show-more-less-html__markup",

        # ----------------------------------------------------
        # Microsoft / career pages
        # ----------------------------------------------------

        "[data-automation-id='jobPostingDescription']",
        "[data-automation-id='jobPostingInfo']",
        "[data-automation-id='jobDescription']",
        "[data-automation-id='job-detail']",
        "[data-automation-id='jobPosting']",
        "[data-testid='job-description']",
        "[data-testid='jobDescription']",

        # ----------------------------------------------------
        # Other common names
        # ----------------------------------------------------

        ".job-content",
        ".job-content-container",
        ".job-posting",
        ".job-posting-content",
        ".posting-content",
        ".job-info",
        ".job-information",
        ".career-detail",
        ".career-details",
        ".job-summary",
        ".job-overview",
        ".position-description",
        ".position-details",
    ]

    candidates = []

    for selector in selectors:

        try:

            elements = soup.select(
                selector
            )

        except Exception:

            continue

        for element in elements:

            text = element.get_text(
                "\n",
                strip=True,
            )

            text = _clean_text(
                text
            )

            if len(text) >= 150:

                candidates.append(
                    text
                )

    # --------------------------------------------------------
    # Search elements by class/id names containing job terms.
    #
    # This helps with sites whose class names are dynamic.
    # --------------------------------------------------------

    dynamic_candidates = []

    for element in soup.find_all(
        [
            "div",
            "section",
            "article",
            "main",
        ]
    ):

        class_value = element.get(
            "class",
            []
        )

        id_value = element.get(
            "id",
            ""
        )

        class_text = " ".join(
            class_value
            if isinstance(
                class_value,
                list,
            )
            else [str(class_value)]
        )

        identifier = (
            f"{class_text} {id_value}"
        ).lower()

        if not any(
            keyword in identifier
            for keyword in [
                "job",
                "posting",
                "description",
                "responsibil",
                "qualif",
                "requirement",
                "career",
                "position",
                "role",
            ]
        ):

            continue

        text = element.get_text(
            "\n",
            strip=True,
        )

        text = _clean_text(
            text
        )

        if len(text) >= 200:

            dynamic_candidates.append(
                text
            )

    candidates.extend(
        dynamic_candidates
    )

    # --------------------------------------------------------
    # FULL BODY FALLBACK
    # --------------------------------------------------------

    body_text = ""

    if soup.body:

        body_text = soup.body.get_text(
            "\n",
            strip=True,
        )

        body_text = _clean_text(
            body_text
        )

    # --------------------------------------------------------
    # Select best candidate.
    # --------------------------------------------------------

    selected_text = ""

    if candidates:

        selected_text = max(
            candidates,
            key=_job_content_score,
        )

    # --------------------------------------------------------
    # Build combined content.
    # --------------------------------------------------------

    parts = []

    if json_ld_text:
        parts.append(
            json_ld_text
        )

    if meta_text:
        parts.append(
            meta_text
        )

    if selected_text:
        parts.append(
            selected_text
        )

    # --------------------------------------------------------
    # Body fallback.
    # --------------------------------------------------------

    if (
        not selected_text
        and body_text
    ):

        parts.append(
            body_text
        )

    combined = _clean_text(
        "\n\n".join(parts)
    )

    return (
        combined,
        title,
    )


# ============================================================
# NORMAL HTTP FETCH
# ============================================================

def _fetch_with_requests(
    url: str,
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

    if response.status_code >= 400:

        raise URLFetchError(
            "The job webpage returned HTTP "
            f"{response.status_code}."
        )

    content_type = (
        response.headers
        .get(
            "Content-Type",
            "",
        )
        .lower()
    )

    if (
        "text/html"
        not in content_type
        and "application/xhtml"
        not in content_type
    ):

        raise URLFetchError(
            "The supplied URL does not appear "
            "to contain an HTML webpage."
        )

    text, title = (
        _extract_text_from_html(
            response.text
        )
    )

    return (
        text,
        title,
        response.url,
    )


# ============================================================
# PLAYWRIGHT / JAVASCRIPT FETCH
# ============================================================

def _fetch_with_playwright(
    url: str,
) -> tuple[str, str, str]:
    """
    Render JavaScript-heavy job pages using Chromium.

    Uses:
        1. Rendered HTML extraction
        2. Direct visible browser text extraction
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
    context = None

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
                ignore_https_errors=True,
            )

            page = context.new_page()

            # ------------------------------------------------
            # Additional browser headers.
            # ------------------------------------------------

            page.set_extra_http_headers(
                {
                    "Accept-Language": (
                        "en-US,en;q=0.9"
                    )
                }
            )

            # ------------------------------------------------
            # Load page.
            # ------------------------------------------------

            page.goto(
                url,
                wait_until="domcontentloaded",
                timeout=30000,
            )

            # ------------------------------------------------
            # Allow JS to finish.
            # ------------------------------------------------

            try:

                page.wait_for_load_state(
                    "networkidle",
                    timeout=12000,
                )

            except PlaywrightTimeoutError:

                pass

            # ------------------------------------------------
            # Extra rendering time.
            # ------------------------------------------------

            page.wait_for_timeout(
                5000
            )

            # ------------------------------------------------
            # Scroll through page.
            # ------------------------------------------------

            try:

                page.evaluate(
                    """
                    async () => {
                        await new Promise((resolve) => {
                            let total = 0;
                            const distance = 400;
                            const maxScrolls = 80;
                            let count = 0;

                            const timer = setInterval(() => {

                                window.scrollBy(
                                    0,
                                    distance
                                );

                                total += distance;
                                count += 1;

                                const height =
                                    document.body
                                    ? document.body.scrollHeight
                                    : 0;

                                if (
                                    total >= height ||
                                    count >= maxScrolls
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
                    2000
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

            # ------------------------------------------------
            # Direct visible text.
            # ------------------------------------------------

            visible_text = ""

            try:

                visible_text = (
                    page.locator(
                        "body"
                    ).inner_text(
                        timeout=7000
                    )
                )

            except Exception:

                try:

                    visible_text = (
                        page.evaluate(
                            """
                            () => document.body
                                ? document.body.innerText
                                : ""
                            """
                        )
                    )

                except Exception:

                    visible_text = ""

            visible_text = _clean_text(
                visible_text
            )

            # ------------------------------------------------
            # Browser diagnostics.
            #
            # These are intentionally printed because we are
            # testing job portals with different structures.
            # ------------------------------------------------

            print(
                "\n========== JOB URL DEBUG =========="
            )

            print(
                "Requested URL:",
                url
            )

            print(
                "Final URL:",
                final_url
            )

            print(
                "Page title:",
                title
            )

            print(
                "Rendered HTML length:",
                len(html or "")
            )

            print(
                "Visible text length:",
                len(visible_text)
            )

            print(
                "Visible job terms:",
                _job_term_count(
                    visible_text
                )
            )

            print(
                "Visible looks like job:",
                _looks_like_job_page(
                    visible_text
                )
            )

            print(
                "Visible preview:",
                visible_text[:1200]
            )

            print(
                "===================================\n"
            )

            # ------------------------------------------------
            # Close resources.
            # ------------------------------------------------

            context.close()
            context = None

            browser.close()
            browser = None

    except PlaywrightTimeoutError as exc:

        if context:

            try:
                context.close()
            except Exception:
                pass

        if browser:

            try:
                browser.close()
            except Exception:
                pass

        raise URLFetchError(
            "The job webpage took too long to load."
        ) from exc

    except Exception as exc:

        if context:

            try:
                context.close()
            except Exception:
                pass

        if browser:

            try:
                browser.close()
            except Exception:
                pass

        raise URLFetchError(
            f"Unable to render the job webpage: {exc}"
        ) from exc

    # ========================================================
    # HTML EXTRACTION
    # ========================================================

    html_text, extracted_title = (
        _extract_text_from_html(
            html
        )
    )

    if extracted_title:
        title = extracted_title

    html_text = _clean_text(
        html_text
    )

    # ========================================================
    # CANDIDATES
    # ========================================================

    candidates = []

    if html_text:
        candidates.append(
            html_text
        )

    if visible_text:
        candidates.append(
            visible_text
        )

    # --------------------------------------------------------
    # Prefer valid job candidates.
    # --------------------------------------------------------

    valid_candidates = [
        candidate
        for candidate in candidates
        if _looks_like_job_page(
            candidate
        )
    ]

    if valid_candidates:

        text = max(
            valid_candidates,
            key=_job_content_score,
        )

    elif candidates:

        # Keep the richest candidate so that the main
        # validation can make the final decision.
        text = max(
            candidates,
            key=_job_content_score,
        )

    else:

        text = ""

    return (
        text,
        title,
        final_url,
    )


# ============================================================
# MAIN FUNCTION
# ============================================================

def fetch_job_page(
    url: str,
) -> dict:
    """
    Fetch a public job webpage and extract readable job text.

    Strategy:

        1. Validate URL.
        2. Try normal HTTP.
        3. Check extracted content.
        4. Fall back to Playwright.
        5. Extract JSON-LD + HTML + visible text.
        6. Validate the result.
        7. Return text and metadata.
    """

    url = _validate_url(
        url
    )

    request_error = None

    # ========================================================
    # STEP 1 — NORMAL HTTP REQUEST
    # ========================================================

    try:

        text, title, final_url = (
            _fetch_with_requests(
                url
            )
        )

        print(
            "\n========== HTTP JOB DEBUG =========="
        )

        print(
            "Requested URL:",
            url
        )

        print(
            "Final URL:",
            final_url
        )

        print(
            "Title:",
            title
        )

        print(
            "Extracted text length:",
            len(text)
        )

        print(
            "Job terms:",
            _job_term_count(
                text
            )
        )

        print(
            "Looks like job:",
            _looks_like_job_page(
                text
            )
        )

        print(
            "====================================\n"
        )

        if _looks_like_job_page(
            text
        ):

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

        print(
            "\nHTTP extraction failed:",
            exc
        )

    # ========================================================
    # STEP 2 — JAVASCRIPT / BROWSER FALLBACK
    # ========================================================

    try:

        text, title, final_url = (
            _fetch_with_playwright(
                url
            )
        )

        # ----------------------------------------------------
        # Final validation.
        # ----------------------------------------------------

        if not _looks_like_job_page(
            text
        ):

            diagnostic = (
                f"Extracted text length: "
                f"{len(text)}; "
                f"job terms: "
                f"{_job_term_count(text)}; "
                f"title: {title!r}; "
                f"final URL: {final_url!r}"
            )

            print(
                "\n========== EXTRACTION FAILED =========="
            )

            print(
                diagnostic
            )

            print(
                "Text preview:",
                text[:1500]
            )

            print(
                "========================================\n"
            )

            raise URLFetchError(
                "The webpage did not contain enough "
                "readable job information after "
                "JavaScript rendering. "
                + diagnostic
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
        # Handle removed / unavailable job postings clearly.
        # ----------------------------------------------------

        error_message = str(browser_error).lower()

        if (
            "http 404" in error_message
            or "404" in error_message
            or "http 410" in error_message
            or "410" in error_message
            or "not found" in error_message
            or "removed" in error_message
            or "closed" in error_message
        ):
            raise URLFetchError(
                "Job posting unavailable. "
                "This job posting has been removed, "
                "closed, or is no longer available. "
                "A reliable scam analysis cannot be "
                "performed because the original job "
                "content is unavailable."
            ) from browser_error

        # ----------------------------------------------------
        # Other extraction failures keep the existing message.
        # ----------------------------------------------------

        raise URLFetchError(
            "Unable to extract enough readable job "
            "information from this webpage. "
            "The site may be JavaScript-rendered, "
            "blocked, or may not contain a public job "
            "description."
        ) from browser_error