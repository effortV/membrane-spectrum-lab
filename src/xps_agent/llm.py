from __future__ import annotations

import base64
import json
import mimetypes
import time
import uuid
from pathlib import Path
from typing import Any, Callable

import httpx
from tenacity import Retrying, stop_after_attempt, wait_exponential_jitter

from .config import Settings
from .db import StateDB
from .security import safe_error
from .tasks import OperationStopped
from .utils import extract_json, json_dumps, sha256_text


class LLMError(RuntimeError):
    pass


class LLMTransientError(LLMError):
    pass


class LLMBudgetExceeded(LLMError):
    pass


class LLMTimeoutError(LLMError):
    pass


class SiliconFlowClient:
    def __init__(
        self,
        settings: Settings,
        db: StateDB | None = None,
        *,
        progress: Callable[..., None] | None = None,
        check_cancelled: Callable[[], None] | None = None,
    ):
        if not settings.siliconflow_api_key:
            raise LLMError("SiliconFlow API key 未配置，请在网页“系统设置”中保存密钥。")
        self.settings = settings
        self.db = db
        self.network_requests = 0
        self.progress = progress or (lambda **_: None)
        self.check_cancelled = check_cancelled or (lambda: None)
        self.client = httpx.Client(
            base_url=settings.siliconflow_base_url,
            timeout=settings.llm_timeout_seconds,
            headers={
                "Authorization": f"Bearer {settings.siliconflow_api_key}",
                "Content-Type": "application/json",
            },
        )

    def _request_hash(self, payload: dict[str, Any]) -> str:
        scrubbed = {key: value for key, value in payload.items() if key != "stream"}
        return sha256_text(json_dumps(scrubbed))

    def _read_stream(self, response: httpx.Response, started: float) -> dict[str, Any]:
        content, reasoning, tool_calls = [], [], {}
        finish_reason = None
        usage = None
        chunks = 0
        content_chars = reasoning_chars = 0
        last_update = -float("inf")
        pending = []

        def consume(data: str) -> bool:
            nonlocal finish_reason, usage, chunks, content_chars, reasoning_chars, last_update
            if data.strip() == "[DONE]":
                return True
            item = json.loads(data)
            if not isinstance(item, dict) or item.get("error"):
                raise LLMError("流式接口返回错误，未保存为成功结果。")
            chunks += 1
            if isinstance(item.get("usage"), dict):
                usage = item["usage"]
            choices = item.get("choices", [])
            if not isinstance(choices, list):
                raise LLMError("流式 choices 格式无效。")
            for choice in choices:
                if not isinstance(choice, dict) or choice.get("index", 0) != 0:
                    raise LLMError("流式返回了非预期的响应分支。")
                delta = choice.get("delta") or {}
                if not isinstance(delta, dict):
                    raise LLMError("流式 delta 格式无效。")
                for key, target in (("content", content), ("reasoning_content", reasoning)):
                    value = delta.get(key)
                    if value is not None:
                        if not isinstance(value, str):
                            raise LLMError("流式文本格式无效。")
                        target.append(value)
                        if key == "content":
                            content_chars += len(value)
                        else:
                            reasoning_chars += len(value)
                calls = delta.get("tool_calls") or []
                if not isinstance(calls, list):
                    raise LLMError("流式工具调用格式无效。")
                for call in calls:
                    if not isinstance(call, dict):
                        raise LLMError("流式工具调用格式无效。")
                    index = call.get("index")
                    if not isinstance(index, int) or not 0 <= index < 8:
                        raise LLMError("流式工具调用超过安全上限。")
                    merged = tool_calls.setdefault(
                        index,
                        {"id": "", "type": "function", "function": {"name": "", "arguments": ""}},
                    )
                    if call.get("id"):
                        if not isinstance(call["id"], str):
                            raise LLMError("流式工具 ID 无效。")
                        if call["id"] != merged["id"]:
                            merged["id"] += call["id"]
                    function = call.get("function") or {}
                    if not isinstance(function, dict):
                        raise LLMError("流式工具函数无效。")
                    for key in ("name", "arguments"):
                        value = function.get(key)
                        if value is not None:
                            if not isinstance(value, str):
                                raise LLMError("流式工具参数无效。")
                            merged["function"][key] += value
                            content_chars += len(value)
                if choice.get("finish_reason"):
                    if not isinstance(choice["finish_reason"], str):
                        raise LLMError("流式结束标记无效。")
                    finish_reason = choice["finish_reason"]
            if content_chars + reasoning_chars > 5_000_000:
                raise LLMError("模型流式响应超过本地安全大小上限。")
            now = time.monotonic()
            if now - last_update >= 1 or finish_reason:
                stage = (
                    "模型正在生成"
                    if content_chars
                    else "模型正在推理"
                    if reasoning_chars
                    else "已连接，等待生成"
                )
                self.progress(
                    stage=stage,
                    received_chunks=chunks,
                    generated_chars=content_chars,
                    reasoning_chars=reasoning_chars,
                    tool_calls_count=len(tool_calls),
                    request_elapsed_seconds=round(now - started, 1),
                )
                last_update = now
            return False

        for line in response.iter_lines():
            self.check_cancelled()
            if time.monotonic() - started >= self.settings.request_timeout_seconds:
                raise LLMTimeoutError("本次模型请求达到总时限，未将不完整响应保存为成功结果。")
            if line.startswith("data:"):
                pending.append(line[5:].lstrip())
                if sum(map(len, pending)) > 1_000_000:
                    raise LLMError("流式事件过大。")
            elif not line and pending:
                done = consume("\n".join(pending))
                pending.clear()
                if done:
                    break
        if pending:
            consume("\n".join(pending))
        if not finish_reason:
            raise LLMTimeoutError("模型连接已结束，但响应未完整返回；未缓存半截结果。")
        message: dict[str, Any] = {"role": "assistant", "content": "".join(content)}
        if reasoning:
            message["reasoning_content"] = "".join(reasoning)
        if tool_calls:
            message["tool_calls"] = [tool_calls[index] for index in sorted(tool_calls)]
        return {
            "object": "chat.completion",
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": usage or {},
        }

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.check_cancelled()
        if self.network_requests >= self.settings.llm_max_requests_per_action:
            raise LLMBudgetExceeded(
                f"已达到本次操作 {self.settings.llm_max_requests_per_action} 次网络请求上限；"
                "已完成的结果已缓存，请分批继续。"
            )
        self.network_requests += 1
        request_hash = self._request_hash(payload)
        model = str(payload["model"])
        wait_seconds = (
            self.settings.vision_timeout_seconds
            if model == self.settings.vision_model
            else self.settings.llm_timeout_seconds
        )
        wait_seconds = min(wait_seconds, self.settings.request_timeout_seconds)
        trace_id = uuid.uuid4().hex
        started = time.monotonic()
        self.progress(
            stage="等待模型响应",
            model=model,
            network_requests=self.network_requests,
            wait_limit_seconds=wait_seconds,
            request_trace_id=trace_id,
            request_total_limit_seconds=self.settings.request_timeout_seconds,
            received_chunks=0,
            generated_chars=0,
            reasoning_chars=0,
            last_error="",
        )
        try:
            with self.client.stream(
                "POST",
                "/chat/completions",
                json=payload,
                headers={"X-Trace-Id": trace_id},
                timeout=httpx.Timeout(wait_seconds, connect=10.0, write=60.0, pool=10.0),
            ) as response:
                if response.status_code in {401, 403}:
                    raise LLMError(
                        "SiliconFlow authentication or model permission failed; not retried"
                    )
                if response.status_code in {408, 429} or response.status_code >= 500:
                    raise LLMTransientError(
                        f"SiliconFlow transient error: HTTP {response.status_code}"
                    )
                if response.status_code >= 400:
                    raise LLMError(
                        f"SiliconFlow HTTP {response.status_code}; check model/request settings"
                    )
                if "text/event-stream" in response.headers.get("content-type", ""):
                    body = self._read_stream(response, started)
                else:
                    response.read()
                    body = response.json()
            if time.monotonic() - started >= self.settings.request_timeout_seconds:
                raise LLMTimeoutError("本次模型请求达到总时限。")
            if not isinstance(body, dict) or not body.get("choices"):
                raise LLMError("SiliconFlow returned no choices")
            if (
                not isinstance(body["choices"], list)
                or not isinstance(body["choices"][0], dict)
                or not isinstance(body["choices"][0].get("message"), dict)
            ):
                raise LLMError("SiliconFlow response has an invalid choices/message schema")
        except (httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
            error = LLMTransientError(f"模型连接建立超时（{type(exc).__name__}）。")
            if self.db:
                self.db.record_llm_event(request_hash, model, "failed", error=safe_error(error))
            raise error from None
        except (
            httpx.TimeoutException,
            LLMTimeoutError,
            httpx.RemoteProtocolError,
            httpx.ReadError,
            httpx.WriteError,
        ) as exc:
            error = LLMTimeoutError(
                f"模型响应中断（{type(exc).__name__}，无数据等待上限 {wait_seconds:g} 秒）。"
                + (f" {str(exc)} " if isinstance(exc, LLMTimeoutError) else "")
                + f"已保留成功结果，可延长等待或重试失败项。请求追踪 ID：{trace_id}。"
                "不能确认供应商是否计费；未开启超时重试时不自动重复提交。"
            )
            if self.db:
                self.db.record_llm_event(request_hash, model, "failed", error=safe_error(error))
            raise error from None
        except OperationStopped as exc:
            if self.db:
                self.db.record_llm_event(request_hash, model, "interrupted", error=safe_error(exc))
            raise
        except httpx.ConnectError as exc:
            error = LLMTransientError(f"SiliconFlow transport error: {type(exc).__name__}")
            if self.db:
                self.db.record_llm_event(request_hash, model, "failed", error=safe_error(error))
            raise error from exc
        except httpx.TransportError as exc:
            error = LLMError(
                f"模型网络连接中断（{type(exc).__name__}），未自动重复提交已发出的请求。"
            )
            if self.db:
                self.db.record_llm_event(request_hash, model, "failed", error=safe_error(error))
            raise error from None
        except (LLMError, ValueError) as exc:
            if self.db:
                self.db.record_llm_event(request_hash, model, "failed", error=safe_error(exc))
            if isinstance(exc, ValueError):
                raise LLMError("SiliconFlow response was not valid JSON") from exc
            raise
        if self.db:
            self.db.record_llm_event(request_hash, model, "succeeded", body.get("usage"))
        self.progress(stage="模型已返回，正在保存", model=model)
        return body

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        model: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        response_format: dict[str, Any] | None = None,
        tools: list[dict[str, Any]] | None = None,
        use_cache: bool = True,
    ) -> dict[str, Any]:
        self.check_cancelled()
        payload: dict[str, Any] = {
            "model": model or self.settings.main_model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": self.settings.llm_stream_enabled,
        }
        if response_format:
            payload["response_format"] = response_format
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        request_hash = self._request_hash(payload)
        if use_cache and self.db:
            cached = self.db.cache_get(request_hash)
            if cached is not None:
                self.db.record_llm_event(request_hash, str(payload["model"]), "cache_hit")
                self.progress(stage="使用已有模型缓存", model=str(payload["model"]))
                return cached
        attempts = (
            self.settings.vision_max_attempts
            if payload["model"] == self.settings.vision_model
            else max(1, self.settings.llm_max_retries)
        )
        timeout_retries = (
            self.settings.vision_timeout_retries
            if payload["model"] == self.settings.vision_model
            else self.settings.llm_timeout_retries
        )
        attempts = max(attempts, timeout_retries + 1)

        def should_retry(state: Any) -> bool:
            nonlocal timeout_retries
            error = state.outcome.exception()
            if isinstance(error, LLMTimeoutError) and timeout_retries > 0:
                timeout_retries -= 1
                return True
            return isinstance(error, LLMTransientError)

        def before_sleep(state: Any) -> None:
            self.check_cancelled()
            self.progress(
                stage="响应超时，按授权准备重试"
                if isinstance(state.outcome.exception(), LLMTimeoutError)
                else "暂时性错误，准备重试",
                attempt=state.attempt_number,
                max_attempts=attempts,
                last_error=safe_error(state.outcome.exception()),
            )

        retrying = Retrying(
            retry=should_retry,
            wait=wait_exponential_jitter(initial=1, max=20),
            stop=stop_after_attempt(attempts),
            before_sleep=before_sleep,
            before=lambda state: self.progress(attempt=state.attempt_number, max_attempts=attempts),
            reraise=True,
        )
        body = retrying(self._post, payload)
        if self.db:
            usage = body.get("usage") or {}
            self.db.cache_put(
                request_hash,
                str(payload["model"]),
                body,
                usage.get("prompt_tokens"),
                usage.get("completion_tokens"),
            )
        return body

    def chat_json(self, messages: list[dict[str, Any]], **kwargs: Any) -> Any:
        kwargs.setdefault("response_format", {"type": "json_object"})
        body = self.chat(messages, **kwargs)
        if body["choices"][0].get("finish_reason") == "length":
            raise LLMError("模型 JSON 响应被长度上限截断；已计入调用记录，不自动重试付费请求。")
        content = body["choices"][0]["message"].get("content") or ""
        try:
            result = extract_json(content)
        except (ValueError, TypeError) as exc:
            raise LLMError("模型未返回有效 JSON；响应已缓存供检查，不自动重试。") from exc
        if not isinstance(result, dict):
            raise LLMError("模型 JSON 顶层必须是对象。")
        return result

    def close(self) -> None:
        self.client.close()

    def __enter__(self) -> "SiliconFlowClient":
        return self

    def __exit__(self, *_: Any) -> None:
        self.close()

    def vision_json(
        self,
        prompt: str,
        image_paths: list[Path],
        *,
        max_tokens: int = 4096,
    ) -> Any:
        content: list[dict[str, Any]] = [{"type": "text", "text": prompt}]
        for path in image_paths:
            mime = mimetypes.guess_type(path.name)[0] or "image/png"
            encoded = base64.b64encode(path.read_bytes()).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{encoded}", "detail": "high"},
                }
            )
        messages = [{"role": "user", "content": content}]
        return self.chat_json(
            messages,
            model=self.settings.vision_model,
            max_tokens=max_tokens,
            temperature=0.0,
        )
