import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


@dataclass
class CallResult:
    text: str
    seconds: float
    input_tokens: int
    output_tokens: int


class RateLimitExhausted(RuntimeError):
    pass


class LLMClient:
    def __init__(self):
        load_dotenv(override=True)

        self.model = os.getenv("CODEX_MODEL", "").strip() or "codex-default"
        self.base_url = "codex-cli-chatgpt-sign-in"
        self.max_tokens = int(os.getenv("CODEX_MAX_TOKENS", "3000"))
        self.timeout = int(os.getenv("CODEX_TIMEOUT", "240"))
        self.calls = []

        configured = os.getenv("CODEX_PATH", "").strip()
        found = configured or shutil.which("codex")
        if not found:
            fallback = Path.home() / ".local" / "bin" / "codex"
            if fallback.is_file():
                found = str(fallback)

        if not found:
            raise RuntimeError(
                "Codex CLI not found. Set CODEX_PATH in .env "
            )

        self.codex_path = str(Path(found).expanduser())
        if not (Path(self.codex_path).is_file() or shutil.which(self.codex_path)):
            raise RuntimeError(f"Invalid CODEX_PATH: {self.codex_path}")

    def ask(self, system, user, task):
        started = time.perf_counter()
        prompt = (
            "SYSTEM INSTRUCTIONS:\n"
            f"{system}\n\n"
            "USER REQUEST:\n"
            f"{user}\n\n"
            "Respond to the user request following the system instructions. "
            "Return only the requested final response. "
            "Do not inspect or modify any local project files or use tools.\n"
        )

        try:
            with tempfile.TemporaryDirectory(prefix="codex_cleaning_") as tmp:
                output_path = Path(tmp) / "reply.txt"
                cmd = [
                    self.codex_path,
                    "exec",
                    "--json",
                    "--sandbox", "read-only",
                    "--skip-git-repo-check",
                    "--output-last-message", str(output_path),
                ]
                if self.model != "codex-default":
                    cmd.extend(["--model", self.model])
                cmd.append("-")

                completed = subprocess.run(
                    cmd,
                    input=prompt,
                    text=True,
                    capture_output=True,
                    cwd=tmp,
                    timeout=self.timeout,
                    check=False,
                )
                if completed.returncode != 0:
                    raise RuntimeError(
                        "Codex CLI failed: "
                        + (completed.stderr.strip()[-1500:] or
                           f"exit code {completed.returncode}")
                    )
                if not output_path.is_file():
                    raise RuntimeError("Codex CLI did not write a final response")
                answer = output_path.read_text(encoding="utf-8")
                if not answer.strip():
                    raise RuntimeError("Codex CLI returned an empty response")

            import json
            token_usage = None
            for line in completed.stdout.splitlines():
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if event.get("type") == "turn.completed" and isinstance(event.get("usage"), dict):
                    token_usage = event["usage"]
            usage_available = token_usage is not None and all(
                type(token_usage.get(k)) is int for k in ("input_tokens", "output_tokens")
            )
            elapsed = time.perf_counter() - started
            self.calls.append({
                "task": task, "seconds": elapsed, "success": True,
                "input_tokens": token_usage["input_tokens"] if usage_available else 0,
                "output_tokens": token_usage["output_tokens"] if usage_available else 0,
                "cached_input_tokens": token_usage.get("cached_input_tokens") if usage_available else None,
                "usage_available": usage_available, "error": "",
            })
            return CallResult(answer, elapsed, self.calls[-1]["input_tokens"], self.calls[-1]["output_tokens"])

        except Exception as exc:
            elapsed = time.perf_counter() - started
            self.calls.append({
                "task": task, "seconds": elapsed, "success": False,
                "input_tokens": 0, "output_tokens": 0,
                "usage_available": False,
                "error": f"{type(exc).__name__}: {exc}",
            })
            raise


def extract_json(text):
    import json
    import re
    text = re.sub(
        r"^\s*```(?:json)?\s*|\s*```\s*$", "",
        text.strip(), flags=re.I
    )
    return json.loads(text)
