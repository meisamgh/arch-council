from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path

IGNORED_DIRS = {
    ".git",
    ".venv",
    "venv",
    "node_modules",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".idea",
    ".vscode",
}

IGNORED_FILES = {
    "package-lock.json",
    "pnpm-lock.yaml",
    "yarn.lock",
    "credentials.json",
    "service-account.json",
}

SECRET_SUFFIXES = {".pem", ".key", ".p12", ".pfx"}

TEXT_SUFFIXES = {
    ".py",
    ".sql",
    ".md",
    ".toml",
    ".yaml",
    ".yml",
    ".json",
    ".txt",
    ".sh",
    ".ts",
    ".tsx",
    ".js",
    ".jsx",
    ".java",
    ".go",
    ".rs",
    ".scala",
    ".kt",
    ".kts",
}

SPECIAL_TEXT_NAMES = {
    "Dockerfile",
    "Makefile",
    "Procfile",
    "requirements.txt",
    "requirements-dev.txt",
}

PRIORITY_NAMES = {
    "README.md": 0,
    "README": 0,
    "README.rst": 0,
    "README.txt": 0,
    "pyproject.toml": 1,
    "requirements.txt": 2,
    "package.json": 3,
    "docker-compose.yml": 4,
    "docker-compose.yaml": 4,
    "Dockerfile": 5,
}

ARCHITECTURE_HINTS = (
    "main",
    "app",
    "config",
    "settings",
    "service",
    "router",
    "pipeline",
    "agent",
    "retriev",
    "model",
    "schema",
    "database",
    "storage",
    "worker",
    "api",
)

README_NAMES = ("README.md", "README", "README.rst", "README.txt")


@dataclass(frozen=True)
class RepositoryContext:
    root: Path
    text: str
    included_files: tuple[str, ...]
    truncated: bool
    mode: str


def _is_allowed(path: Path) -> bool:
    lower_name = path.name.lower()
    if lower_name.startswith(".env"):
        return False
    if path.name in IGNORED_FILES or path.suffix.lower() in SECRET_SUFFIXES:
        return False
    if any(part in IGNORED_DIRS for part in path.parts):
        return False
    if path.name in SPECIAL_TEXT_NAMES:
        return True
    return path.suffix.lower() in TEXT_SUFFIXES


def _priority(path: Path) -> tuple[int, int, int, str]:
    lower = path.as_posix().lower()
    architecture_score = 0 if any(hint in lower for hint in ARCHITECTURE_HINTS) else 1
    return (
        PRIORITY_NAMES.get(path.name, 20),
        architecture_score,
        len(path.parts),
        lower,
    )


def _safe_read(path: Path, max_file_chars: int = 30_000) -> str | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError):
        return None
    if "\x00" in raw:
        return None
    if len(raw) > max_file_chars:
        return raw[:max_file_chars] + "\n... [file truncated] ...\n"
    return raw


def _resolve_root(repo_path: str | Path) -> Path:
    root = Path(repo_path).expanduser().resolve()
    if not root.is_dir():
        raise ValueError(f"Repository path does not exist or is not a directory: {root}")
    return root


def build_readme_context(
    repo_path: str | Path,
    *,
    max_chars: int = 120_000,
) -> RepositoryContext:
    """Build context from only the repository's root README file.

    This is intentionally strict: it does not search subdirectories or append the
    repository tree. It is useful when the council should reason from the project's
    stated architecture rather than inspect implementation details.
    """
    root = _resolve_root(repo_path)
    if max_chars < 1_000:
        raise ValueError("max_chars must be at least 1000")

    readme: Path | None = None
    for name in README_NAMES:
        candidate = root / name
        if candidate.is_file():
            readme = candidate
            break

    if readme is None:
        candidates = {p.name.lower(): p for p in root.iterdir() if p.is_file()}
        readme = next((candidates[name.lower()] for name in README_NAMES
                       if name.lower() in candidates), None)
    if readme is None:
        expected = ", ".join(README_NAMES)
        raise ValueError(f"--readme-only could not find a root README ({expected}) in: {root}")

    try:
        content = readme.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as exc:
        raise ValueError(f"Could not read {readme.name}: {exc}") from exc

    truncated = len(content) > max_chars
    if truncated:
        content = content[:max_chars] + "\n... [README truncated] ...\n"

    text = f"# README-only repository evidence\n\n--- FILE: {readme.name} ---\n{content}\n--- END FILE ---\n"
    return RepositoryContext(
        root=root,
        text=text,
        included_files=(readme.name,),
        truncated=truncated,
        mode="readme-only",
    )


def build_repository_context(
    repo_path: str | Path,
    *,
    max_chars: int = 120_000,
    max_file_chars: int = 30_000,
) -> RepositoryContext:
    root = _resolve_root(repo_path)
    if max_chars < 1_000:
        raise ValueError("max_chars must be at least 1000")

    candidates = sorted(
        (p for p in root.rglob("*") if p.is_file() and _is_allowed(p.relative_to(root))),
        key=lambda p: _priority(p.relative_to(root)),
    )

    tree_lines = [p.relative_to(root).as_posix() for p in candidates]
    tree_budget = min(max_chars // 4, 30_000)
    tree = "\n".join(tree_lines)
    tree_truncated = len(tree) > tree_budget
    if tree_truncated:
        tree = tree[:tree_budget] + "\n... [tree truncated] ..."

    prefix = "# Repository tree\n" + tree + "\n\n# File contents\n"
    remaining = max_chars - len(prefix)
    chunks: list[str] = [prefix]
    included: list[str] = []
    truncated = tree_truncated

    for path in candidates:
        relative = path.relative_to(root).as_posix()
        content = _safe_read(path, max_file_chars=max_file_chars)
        if content is None:
            continue
        block = f"\n--- FILE: {relative} ---\n{content}\n--- END FILE ---\n"
        if len(block) > remaining:
            truncated = True
            continue
        chunks.append(block)
        included.append(relative)
        remaining -= len(block)

    return RepositoryContext(
        root=root,
        text="".join(chunks),
        included_files=tuple(included),
        truncated=truncated,
        mode="repository",
    )


def build_diff_context(
    repo_path: str | Path,
    base: str,
    *,
    max_chars: int = 120_000,
) -> RepositoryContext:
    root = _resolve_root(repo_path)
    if not (root / ".git").exists():
        raise ValueError(f"--diff-base requires a Git repository: {root}")
    if max_chars < 1_000:
        raise ValueError("max_chars must be at least 1000")

    command = ["git", "-C", str(root), "diff", "--no-ext-diff", f"{base}...HEAD"]
    result = subprocess.run(command, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise RuntimeError(f"git diff failed: {result.stderr.strip()}")

    diff = result.stdout
    truncated = len(diff) > max_chars
    if truncated:
        diff = diff[:max_chars] + "\n... [diff truncated] ...\n"

    changed = []
    for line in result.stdout.splitlines():
        if line.startswith("+++ b/"):
            changed.append(line[6:])

    text = f"# Git diff\nBase: {base}\nHead: HEAD\n\n{diff}"
    return RepositoryContext(
        root=root,
        text=text,
        included_files=tuple(dict.fromkeys(changed)),
        truncated=truncated,
        mode=f"diff:{base}",
    )
