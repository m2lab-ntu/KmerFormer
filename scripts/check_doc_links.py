#!/usr/bin/env python3
"""Check every relative link and heading anchor in the repository's markdown.

The anchor half has to reproduce GitHub's slug algorithm exactly, and the obvious
approximation does not. `github-slugger` lowercases, deletes every character that
is not a word character, whitespace or a hyphen, and then replaces each remaining
space with one hyphen. It does *not* collapse runs of whitespace. So a heading
punctuated with an em-dash

    ## 3. Part 3 — at 13-mer the vocabulary stops being free

loses the dash and keeps the spaces that surrounded it, giving a **double**
hyphen:

    3-part-3--at-13-mer-the-vocabulary-stops-being-free

An earlier version of this check collapsed whitespace and therefore computed the
single-hyphen form. It reported such links as valid while GitHub served a 404,
which is worse than not checking: the failure is invisible on a local clone and
only appears on the published page.

Usage:

    python scripts/check_doc_links.py            # whole repo, exit 1 on any break
    python scripts/check_doc_links.py -v         # also list what passed
"""

import argparse
import re
import subprocess
import sys
import urllib.parse
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Directories whose markdown is not published with the repo.
SKIP_DIRS = {"agent_prompts"}

ATX_HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*$")
FENCE = re.compile(r"^\s*(```|~~~)")
MD_LINK = re.compile(r"\[(?:[^\]]*)\]\(([^)\s]+)\)")
# Inline markup GitHub strips before slugging, plus link syntax in headings.
INLINE = re.compile(r"`|\*\*|\*|__|_|~~")
HEADING_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")


def markdown_files(root):
    """Check source documents, excluding ignored local delivery snapshots."""
    if (root / ".git").exists():
        result = subprocess.run(
            ["git", "ls-files", "--cached", "--others", "--exclude-standard", "-z", "*.md"],
            cwd=root, check=True, capture_output=True)
        files = {root / p.decode() for p in result.stdout.split(b"\0") if p}
    else:
        # Source archives have no Git metadata and contain only delivered files.
        files = set(root.rglob("*.md"))
    return [p for p in sorted(files) if p.is_file()
            and not (SKIP_DIRS & set(p.relative_to(root).parts))]


def github_slug(title: str) -> str:
    """Reproduce github-slugger for a heading's text."""
    text = HEADING_LINK.sub(r"\1", title)
    text = INLINE.sub("", text)
    text = text.lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return text.replace(" ", "-")


def headings(path: Path) -> set:
    """Slugs GitHub would mint for one file, deduplicated the way it does."""
    out, seen, in_fence = set(), {}, False
    for line in path.read_text(errors="replace").splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = ATX_HEADING.match(line)
        if not m:
            continue
        base = github_slug(m.group(2))
        # a repeated heading gets -1, -2, ... appended
        n = seen.get(base, 0)
        seen[base] = n + 1
        out.add(base if n == 0 else f"{base}-{n}")
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    files = markdown_files(REPO)
    slug_cache = {}
    broken, checked = [], 0

    for md in files:
        for link in MD_LINK.findall(md.read_text(errors="replace")):
            if link.startswith(("http://", "https://", "mailto:")):
                continue
            target, _, frag = link.partition("#")
            target = urllib.parse.unquote(target)
            rel = md.relative_to(REPO)

            path = md if not target else (md.parent / target)
            if target:
                # a glob or placeholder in prose is not a link to resolve
                if any(c in target for c in "{}*"):
                    continue
                if not path.exists():
                    broken.append(f"{rel} -> {link}   (no such file)")
                    continue
            checked += 1

            if frag:
                if path.suffix != ".md" or not path.exists():
                    continue
                key = path.resolve()
                if key not in slug_cache:
                    slug_cache[key] = headings(path)
                if frag not in slug_cache[key]:
                    near = [s for s in slug_cache[key] if frag.replace("--", "-") == s
                            or s.replace("--", "-") == frag]
                    hint = f"   (did you mean #{near[0]}?)" if near else ""
                    broken.append(f"{rel} -> {link}   (no such anchor){hint}")
                elif args.verbose:
                    print(f"  ok  {rel} -> {link}")

    print(f"markdown files: {len(files)}   links checked: {checked}")
    if broken:
        print(f"\nbroken: {len(broken)}")
        for b in broken:
            print(f"  {b}")
        return 1
    print("no broken links or anchors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
