"""URL normalization for label-free source matching in fact verification.

Ensures robust, reproducible alignment between retrieved passage URLs
and gold evidence annotations without leaking content or altering semantics.
"""
import re
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


TRACKING_PARAMS = {
    "fbclid", "gclid", "ref", "source", "_ga", "mc_cid", "mc_eid",
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content", "utm_name"
}


def normalize_url(url: str) -> str:
    """Normalize a URL for canonical matching.
    
    Rules applied:
    1. Strip leading and trailing whitespace.
    2. Lowercase scheme and netloc; remove default ports (:80, :443).
    3. Normalize http and https to https for uniform web source comparison.
    4. Remove leading 'www.' from hostname.
    5. Clean path: collapse multiple slashes, lowercase path, strip trailing slash.
    6. Filter out tracking query parameters (utm_*, fbclid, etc.) and sort remaining params.
    7. Discard fragment (#...) entirely.
    """
    if not isinstance(url, str):
        return ""
    url = url.strip()
    if not url:
        return ""

    try:
        parts = urlsplit(url)
    except Exception:
        return url.strip().lower()

    scheme = parts.scheme.lower()
    if scheme in {"http", "https"}:
        scheme = "https"
    elif not scheme:
        scheme = "https"

    netloc = parts.netloc.lower()
    # Strip port if default
    if netloc.endswith(":80"):
        netloc = netloc[:-3]
    elif netloc.endswith(":443"):
        netloc = netloc[:-4]

    # Strip leading www.
    if netloc.startswith("www."):
        netloc = netloc[4:]
    netloc = netloc.rstrip(".")

    # Path cleanup
    path = parts.path
    # Collapse multiple slashes
    path = re.sub(r"/+", "/", path)
    # Lowercase path
    path = path.lower()
    # Strip trailing slash
    path = path.rstrip("/")


    # Query params cleanup
    query_items = parse_qsl(parts.query, keep_blank_values=False)
    filtered_params = []
    for k, v in query_items:
        k_lower = k.lower()
        if k_lower in TRACKING_PARAMS or k_lower.startswith("utm_"):
            continue
        filtered_params.append((k, v))
    filtered_params.sort(key=lambda x: (x[0], x[1]))
    clean_query = urlencode(filtered_params)

    # Discard fragment
    clean_fragment = ""

    return urlunsplit((scheme, netloc, path, clean_query, clean_fragment))


def urls_match(url1: str, url2: str) -> bool:
    """Check if two URLs refer to the same canonical source."""
    norm1 = normalize_url(url1)
    norm2 = normalize_url(url2)
    return bool(norm1 and norm2 and norm1 == norm2)
