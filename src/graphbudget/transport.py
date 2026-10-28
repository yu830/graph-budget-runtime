"""Per-client HTTP proxy policy and optional request-stage timings."""
from __future__ import annotations

import http.client
import ipaddress
import os
import socket
import time
import urllib.parse
import urllib.request


def _loopback_host(host):
    if host is None:
        return False
    if host.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(host) in {
            ipaddress.ip_address("127.0.0.1"), ipaddress.ip_address("::1")
        }
    except ValueError:
        return False


def boot_id():
    try:
        with open("/proc/sys/kernel/random/boot_id", encoding="ascii") as handle:
            return handle.read().strip()
    except OSError:
        return None


def peer_category(response, target_url):
    """Classify the connected peer without retaining an address or credentials."""
    try:
        address = response.fp.raw._sock.getpeername()
        ip = ipaddress.ip_address(address[0].split("%", 1)[0])
        target = urllib.parse.urlparse(target_url)
        expected_port = target.port or (443 if target.scheme == "https" else 80)
        if ip.is_loopback:
            return "target_loopback_port" if address[1] == expected_port else "other_loopback_port"
        return "non_loopback"
    except (AttributeError, OSError, ValueError, TypeError, IndexError):
        return "UNKNOWN"


class _TimedHTTPConnection(http.client.HTTPConnection):
    def __init__(self, *args, trace, target_url, **kwargs):
        super().__init__(*args, **kwargs)
        self.trace = trace
        self.target_url = target_url
        self._create_connection = self._timed_create_connection

    def _mark(self, key):
        self.trace[key] = time.perf_counter_ns()

    def _timed_create_connection(self, address, timeout=socket._GLOBAL_DEFAULT_TIMEOUT,
                                 source_address=None):
        host, port = address
        self._mark("dns_begin_ns")
        try:
            addresses = socket.getaddrinfo(host, port, 0, socket.SOCK_STREAM)
        finally:
            self._mark("dns_end_ns")
        self._mark("connect_begin_ns")
        last_error = None
        for family, kind, proto, _, sockaddr in addresses:
            sock = None
            try:
                sock = socket.socket(family, kind, proto)
                if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
                    sock.settimeout(timeout)
                if source_address:
                    sock.bind(source_address)
                sock.connect(sockaddr)
                self._mark("connect_end_ns")
                ip = ipaddress.ip_address(sockaddr[0].split("%", 1)[0])
                target = urllib.parse.urlparse(self.target_url)
                expected_port = target.port or 80
                self.trace["peer_category"] = (
                    "target_loopback_port" if ip.is_loopback and sockaddr[1] == expected_port
                    else "other_loopback_port" if ip.is_loopback else "non_loopback")
                return sock
            except OSError as error:
                last_error = error
                if sock:
                    sock.close()
        self._mark("connect_end_ns")
        if last_error is not None:
            raise last_error
        raise OSError("getaddrinfo returned no TCP addresses")

    def endheaders(self, message_body=None, *, encode_chunked=False):
        super().endheaders(message_body, encode_chunked=encode_chunked)
        self._mark("request_write_complete_ns")

    def getresponse(self):
        response = super().getresponse()
        self._mark("response_headers_complete_ns")
        return response


class _TimedHTTPHandler(urllib.request.HTTPHandler):
    def __init__(self, trace, target_url):
        super().__init__()
        self.trace = trace
        self.target_url = target_url

    def http_open(self, req):
        def connection(host, **kwargs):
            return _TimedHTTPConnection(host, trace=self.trace, target_url=self.target_url, **kwargs)
        return self.do_open(connection, req)


class HTTPTransport:
    """Use an explicit opener. Auto selects direct only for loopback hosts."""
    def __init__(self, url, mode="auto"):
        parsed = urllib.parse.urlparse(url)
        if mode not in {"auto", "direct", "inherit"}:
            raise ValueError("transport must be auto, direct, or inherit")
        self.url = url
        self.loopback = _loopback_host(parsed.hostname)
        self.mode = ("direct" if self.loopback else "inherit") if mode == "auto" else mode
        handler = urllib.request.ProxyHandler({}) if self.mode == "direct" else urllib.request.ProxyHandler()
        self.proxies = dict(handler.proxies)
        self.opener = urllib.request.build_opener(handler)
        self.metadata = {
            "requested_mode": mode,
            "effective_mode": self.mode,
            "target_class": "loopback" if self.loopback else "other",
            "proxy_configured_for_scheme": bool(self.proxies.get(parsed.scheme)),
            "loopback_bypass": bool(urllib.request.proxy_bypass(parsed.hostname or "")) if self.loopback else None,
            "proxy_environment_present": {
                name: bool(os.environ.get(name))
                for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "no_proxy", "NO_PROXY")
            },
            "peer_classification": "per-request socket observation, UNKNOWN if unavailable",
        }

    def open(self, request, *, timeout, trace=None):
        if trace is None:
            return self.opener.open(request, timeout=timeout)
        parsed = urllib.parse.urlparse(self.url)
        if not self.loopback or parsed.scheme != "http":
            raise ValueError("request-stage diagnostics support local HTTP loopback only")
        handler = urllib.request.ProxyHandler(self.proxies)
        opener = urllib.request.build_opener(handler, _TimedHTTPHandler(trace, self.url))
        return opener.open(request, timeout=timeout)
