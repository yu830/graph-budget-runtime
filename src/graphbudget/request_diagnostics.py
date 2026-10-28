"""Opt-in request-level diagnostics for the pinned local SGLang HTTP server."""
from __future__ import annotations

import json
import os
import time

from .transport import boot_id


def install(path):
    # This is installed only in diagnostic runs, before the ASGI app starts.
    from sglang.srt.entrypoints import http_server
    from sglang.srt.managers.tokenizer_manager import TokenizerManager

    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    boot = boot_id()

    def mark(rid, phase):
        if not isinstance(rid, str) or not rid:
            return
        row = {"request_id": rid, "phase": phase, "perf_counter_ns": time.perf_counter_ns(),
               "boot_id": boot, "pid": os.getpid()}
        os.write(fd, (json.dumps(row, separators=(",", ":")) + "\n").encode())

    class RequestPhases:
        def __init__(self, app):
            self.app = app

        async def __call__(self, scope, receive, send):
            if scope.get("type") != "http" or scope.get("path") != "/generate":
                return await self.app(scope, receive, send)
            header = next((v for k, v in scope.get("headers", [])
                           if k.lower() == b"x-graphbudget-request-id"), b"")
            rid = header.decode("utf-8", errors="replace")
            if not rid:
                return await self.app(scope, receive, send)
            mark(rid, "asgi_entry")
            first_output = False

            async def recorded_receive():
                message = await receive()
                if message.get("type") == "http.request" and not message.get("more_body", False):
                    mark(rid, "body_complete")
                return message

            async def recorded_send(message):
                nonlocal first_output
                if message.get("type") == "http.response.start":
                    mark(rid, "response_start")
                elif message.get("type") == "http.response.body":
                    body = message.get("body", b"")
                    if not first_output and b"output_ids" in body:
                        first_output = True
                        mark(rid, "asgi_first_output")
                await send(message)
                if message.get("type") == "http.response.body" and not message.get("more_body", False):
                    mark(rid, "asgi_response_complete")

            try:
                await self.app(scope, recorded_receive, recorded_send)
            finally:
                mark(rid, "asgi_exit")

    original_generate = TokenizerManager.generate_request
    original_send = TokenizerManager._send_one_request

    async def generate(self, obj, request=None):
        rid = getattr(obj, "rid", None)
        mark(rid, "tokenizer_entry")
        first = True
        try:
            async for response in original_generate(self, obj, request):
                if first:
                    mark(rid, "tokenizer_first_output")
                    first = False
                yield response
        finally:
            mark(rid, "tokenizer_exit")

    def send_one(self, obj, tokenized_obj, created_time=None):
        rid = getattr(obj, "rid", None)
        mark(rid, "scheduler_dispatch_begin")
        result = original_send(self, obj, tokenized_obj, created_time)
        mark(rid, "scheduler_dispatch_return")
        return result

    TokenizerManager.generate_request = generate
    TokenizerManager._send_one_request = send_one
    http_server.app.add_middleware(RequestPhases)
