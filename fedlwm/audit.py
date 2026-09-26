from __future__ import annotations

import re
from pathlib import Path


TEXT_SUFFIXES = {".py", ".md", ".json", ".toml", ".txt", ".sh", ".yaml", ".yml"}


def audit_release(root: str | Path) -> list[str]:
    root = Path(root).resolve()
    findings: list[str] = []
    forbidden = ["file:" + "//"]
    email = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
    posix_home = re.escape("/" + "home" + "/") + r"[^/\s]+/"
    macos_home = re.escape("/" + "Users" + "/") + r"[^/\s]+/"
    windows_home = r"[A-Za-z]:\\" + "Users" + r"\\[^\\\s]+\\"
    machine_path = re.compile(
        "(?:" + posix_home + "|" + macos_home + "|" + windows_home + ")"
    )
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        if any(part in {"runs", "data", "__pycache__", ".pytest_cache"} for part in path.parts):
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for token in forbidden:
            if token.lower() in text.lower():
                findings.append(f"{path.relative_to(root)}: forbidden identity/path token")
        if machine_path.search(text):
            findings.append(f"{path.relative_to(root)}: machine-specific path")
        if email.search(text):
            findings.append(f"{path.relative_to(root)}: email-like identity token")
    return findings
