from urllib.parse import urlparse


def is_url(url: str) -> bool:
    """Check if the given string is a URL."""
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc, result.path])
    except Exception:
        return False
