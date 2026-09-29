from pathlib import Path

from arch_council.context import build_repository_context


def test_repository_context_excludes_env_and_binary_like_files(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("hello", encoding="utf-8")
    (tmp_path / "app.py").write_text("print('ok')", encoding="utf-8")
    (tmp_path / ".env").write_text("SECRET=do-not-send", encoding="utf-8")
    (tmp_path / ".env.local").write_text("OTHER_SECRET=do-not-send", encoding="utf-8")
    (tmp_path / "private.key").write_text("secret-key", encoding="utf-8")
    (tmp_path / "image.png").write_bytes(b"\x89PNG")

    context = build_repository_context(tmp_path)

    assert "README.md" in context.included_files
    assert "app.py" in context.included_files
    assert ".env" not in context.included_files
    assert ".env.local" not in context.included_files
    assert "private.key" not in context.included_files
    assert "do-not-send" not in context.text
    assert "secret-key" not in context.text
    assert "image.png" not in context.included_files


def test_repository_context_honors_size_limit(tmp_path: Path) -> None:
    (tmp_path / "README.md").write_text("x" * 4000, encoding="utf-8")
    context = build_repository_context(tmp_path, max_chars=1000)

    assert context.truncated is True
    assert len(context.text) <= 1000
