#!/usr/bin/env python3
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


RE_OUTPUT_T = re.compile(r"(?:^|[^a-z0-9])output[_-]?t0\.(\d+)(?:[^a-z0-9]|$)")


def is_audio_file(p: Path) -> bool:
    return p.is_file() and p.suffix.lower() in {".wav", ".mp3", ".flac", ".ogg", ".m4a"}


def has_audio_files(d: Path) -> bool:
    return any(is_audio_file(p) for p in d.iterdir())


def audio_child_dirs(d: Path) -> list[Path]:
    out: list[Path] = []
    for c in d.iterdir():
        if not c.is_dir():
            continue
        if any(has_audio_files(x) for x in c.rglob("*") if x.is_dir()) or has_audio_files(c):
            out.append(c)
    return out


def find_leaf_audio_dirs(section_dir: Path) -> list[Path]:
    candidates = [d for d in section_dir.rglob("*") if d.is_dir() and has_audio_files(d)]
    leaves: list[Path] = []
    for d in candidates:
        if any((c.is_dir() and has_audio_files(c)) for c in d.iterdir()):
            continue
        leaves.append(d)
    return sorted(leaves)


def score_output_file(name_lower: str) -> tuple[int, int, str]:
    """
    Higher score should win (we'll sort reverse).
    Preference: output_t0.80 if present; otherwise highest t; otherwise generic output.
    """
    if ".ds_store" in name_lower:
        return (-10, 0, name_lower)
    m = RE_OUTPUT_T.search(name_lower)
    if m:
        t = int(m.group(1))
        preferred = 1 if t == 80 else 0
        return (preferred, t, name_lower)
    if "output" in name_lower:
        return (0, 0, name_lower)
    return (-1, 0, name_lower)


def pick_first(files: list[Path], contains_any: Iterable[str]) -> Optional[Path]:
    keys = tuple(k.lower() for k in contains_any)
    for f in files:
        n = f.name.lower()
        if n == ".ds_store":
            continue
        if any(k in n for k in keys):
            return f
    return None


@dataclass(frozen=True)
class Row:
    label: str
    folder: str
    input: Optional[str]
    output: Optional[str]
    target: Optional[str]
    extras: dict[str, list[str]]


def build_row(section_dir: Path, leaf_dir: Path, repo_root: Path) -> Row:
    rel_from_section = leaf_dir.relative_to(section_dir).as_posix()
    label = rel_from_section

    audio_files = sorted([p for p in leaf_dir.iterdir() if is_audio_file(p) and p.name != ".DS_Store"])

    input_f = pick_first(audio_files, ["input_audio_snippet", "mixture", "input"])

    # target / reference
    target_f = pick_first(audio_files, ["target", "clean", "no_noise", "humanrecording", "groundtruth"])

    # output selection
    output_candidates = [p for p in audio_files if "output" in p.name.lower()]
    output_f: Optional[Path] = None
    if output_candidates:
        output_f = sorted(output_candidates, key=lambda p: score_output_file(p.name.lower()), reverse=True)[0]
    else:
        # fallback: common baselines in some folders
        output_f = pick_first(audio_files, ["demucs", "diffwave", "mel2mel", "gl"])

    # extras: keep remaining outputs/baselines for optional expansion
    extras: dict[str, list[str]] = {}
    remaining = [p for p in audio_files if p not in {input_f, target_f, output_f}]
    extra_outputs = [p for p in remaining if any(k in p.name.lower() for k in ["output", "demucs", "diffwave", "mel2mel", "gl"])]
    other = [p for p in remaining if p not in set(extra_outputs)]
    if extra_outputs:
        extras["more_outputs"] = [p.relative_to(repo_root).as_posix() for p in sorted(extra_outputs)]
    if other:
        extras["other_audio"] = [p.relative_to(repo_root).as_posix() for p in sorted(other)]

    def to_repo_path(p: Optional[Path]) -> Optional[str]:
        return None if p is None else p.relative_to(repo_root).as_posix()

    return Row(
        label=label,
        folder=leaf_dir.relative_to(repo_root).as_posix(),
        input=to_repo_path(input_f),
        output=to_repo_path(output_f),
        target=to_repo_path(target_f),
        extras=extras,
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    audio_root = repo_root / "assets" / "audio"
    if not audio_root.exists():
        raise SystemExit(f"Missing audio root: {audio_root}")

    sections = []
    preferred_order = ["TimbreTransfer", "Midi2Real", "Enhancing", "Limitations"]
    order_index = {name: i for i, name in enumerate(preferred_order)}
    section_dirs = [d for d in audio_root.iterdir() if d.is_dir()]
    section_dirs.sort(key=lambda d: (order_index.get(d.name, 10_000), d.name.lower()))
    for section_dir in section_dirs:
        leaf_dirs = find_leaf_audio_dirs(section_dir)
        rows = [build_row(section_dir, d, repo_root) for d in leaf_dirs]
        sections.append(
            {
                "title": section_dir.name,
                "folder": section_dir.relative_to(repo_root).as_posix(),
                "rows": [
                    {
                        "label": r.label,
                        "folder": r.folder,
                        "input": r.input,
                        "output": r.output,
                        "target": r.target,
                        "extras": r.extras,
                    }
                    for r in rows
                ],
            }
        )

    manifest = {"version": 1, "sections": sections}
    out_path = audio_root / "manifest.json"
    out_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {out_path.relative_to(repo_root)} with {sum(len(s['rows']) for s in sections)} rows.")


if __name__ == "__main__":
    main()

