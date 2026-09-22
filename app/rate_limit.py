from slowapi import Limiter


def get_client_ip(request) -> str:
    """
    Resolves the real client IP for rate-limit keying and audit-log attribution.
    Behind a reverse proxy, request.client.host is the proxy's own address, not
    the visitor's, so the leftmost X-Forwarded-For entry is used instead when
    present.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"


limiter = Limiter(key_func=get_client_ip)
