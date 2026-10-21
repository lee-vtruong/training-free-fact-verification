import pytest
from tffv.retrieval.url_norm import normalize_url, urls_match


def test_url_normalization_scheme_and_www():
    assert normalize_url("http://www.example.com") == "https://example.com"
    assert normalize_url("https://example.com/") == "https://example.com"
    assert normalize_url("http://EXAMPLE.COM/PATH") == "https://example.com/path"
    assert urls_match("http://www.bbc.com/news/123", "https://bbc.com/news/123/") is True


def test_url_normalization_ports_and_slashes():
    assert normalize_url("http://example.com:80/a//b/c/") == "https://example.com/a/b/c"
    assert normalize_url("https://example.com:443/test") == "https://example.com/test"


def test_url_normalization_tracking_params_and_fragments():
    u1 = "https://www.forbes.com/article/?utm_source=twitter&utm_medium=social&id=42#header"
    u2 = "http://forbes.com/article?id=42"
    assert normalize_url(u1) == "https://forbes.com/article?id=42"
    assert urls_match(u1, u2) is True


def test_url_normalization_query_sorting():
    u1 = "https://example.org/search?q=apple&lang=en"
    u2 = "https://example.org/search?lang=en&q=apple"
    assert normalize_url(u1) == normalize_url(u2)
    assert urls_match(u1, u2) is True


def test_url_normalization_edge_cases():
    assert normalize_url("") == ""
    assert normalize_url("   ") == ""
    assert normalize_url(None) == ""
    assert urls_match("", "https://example.com") is False
