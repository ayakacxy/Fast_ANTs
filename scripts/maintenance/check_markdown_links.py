"""Check local Markdown links and English/Chinese navigation."""

import argparse
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit

PROJECT = Path(__file__).resolve().parents[2]
DEFAULTS = ("README.md", "README.zh-CN.md", ".github", "licenses",
            "docs/en", "docs/zh-CN", "configs",
            "validation/README.md", "validation/README.zh-CN.md", "templates/README.md",
            "templates/README.zh-CN.md",
            "tests/README.md", "tests/README.zh-CN.md")


def language(path):
    """Return the language and counterpart of a maintained Markdown page."""
    path = path.resolve()
    if path.suffix != ".md":
        return None, None
    for current, other in (("en", "zh-CN"), ("zh-CN", "en")):
        directory = PROJECT / "docs" / current
        if path.is_relative_to(directory):
            return current, PROJECT / "docs" / other / path.relative_to(directory)
    if path.name.endswith(".zh-CN.md"):
        counterpart = path.with_name(path.name.replace(".zh-CN.md", ".md"))
        return ("zh-CN", counterpart) if counterpart.exists() else (None, None)
    counterpart = path.with_name(path.stem + ".zh-CN.md")
    return ("en", counterpart) if counterpart.exists() else (None, None)


def documents(paths):
    for entry in paths:
        entry = entry.resolve()
        yield from sorted(entry.rglob("*.md")) if entry.is_dir() else [entry]


def check(paths):
    failures, checked = [], 0
    for path in documents(paths):
        if not path.is_file():
            failures.append((str(path), "Document does not exist"))
            continue
        content = re.sub(r"```.*?```", "", path.read_text(), flags=re.S)
        for match in re.finditer(r"(?<!!)\[[^\]\n]+\]\((<[^>]+>|[^)\n]+)\)", content):
            link = match.group(1).strip().strip("<>")
            parts = urlsplit(link)
            if parts.scheme or parts.netloc or not parts.path:
                continue
            target = Path(unquote(parts.path))
            if not target.is_absolute():
                target = path.parent / target
            checked += 1
            if not target.exists():
                failures.append((str(path.relative_to(PROJECT)), link))
    return checked, failures


def check_languages(paths):
    checked, failures = 0, []
    switches = {"English", "简体中文", "中文"}
    for path in documents(paths):
        if not path.is_file():
            continue
        current, counterpart = language(path)
        if current is None:
            continue
        content = re.sub(r"```.*?```", "", path.read_text(), flags=re.S)
        for match in re.finditer(r"(?<!!)\[([^\]\n]+)\]\((<[^>]+>|[^)\n]+)\)", content):
            label, link = match.groups()
            parts = urlsplit(link.strip().strip("<>"))
            if parts.scheme or parts.netloc or not parts.path:
                continue
            target = Path(unquote(parts.path))
            if not target.is_absolute():
                target = path.parent / target
            target_language, _ = language(target)
            if target_language is None:
                continue
            checked += 1
            if label not in switches and target_language != current:
                failures.append((str(path.relative_to(PROJECT)), link))
        if not counterpart.is_file():
            failures.append((str(path.relative_to(PROJECT)), "Missing translation"))
        if not re.search(r"\[English\]\([^\n]+\).*\[简体中文\]\(", content):
            failures.append((str(path.relative_to(PROJECT)), "Missing language selector"))
    return checked, failures


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", type=Path, nargs="*")
    args = parser.parse_args(argv)
    paths = args.paths or [PROJECT / name for name in DEFAULTS]
    checked, failures = check(paths)
    for path, link in failures:
        print(path + ": missing local target: " + link)
    print(f"Checked {checked} local targets; {len(failures)} missing")
    language_links, language_failures = check_languages(paths)
    for path, link in language_failures:
        print(path + ": incorrect language navigation: " + link)
    print(f"Checked {language_links} language links; {len(language_failures)} mismatches")
    return bool(failures or language_failures)


if __name__ == "__main__":
    raise SystemExit(main())
