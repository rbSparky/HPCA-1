#!/usr/bin/env python3
"""Fail-closed preflight for the HPCA-1 anonymous submission manuscript."""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import tempfile
from pathlib import Path


APPROVED_AI_DISCLOSURE = (
    r"\section{AI Use}" + "\n"
    "OpenAI ChatGPT was used for editorial restructuring, LaTeX assistance, and "
    "generation of the two conceptual schematic figures in this revision. It was "
    "not used to generate experimental measurements. The authors remain responsible "
    "for verifying all technical claims, equations, citations, source changes, and "
    "reported results before submission."
)


def run(*argv: str, cwd: Path) -> str:
    proc = subprocess.run(
        argv,
        cwd=cwd,
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(argv)} failed ({proc.returncode}):\n{proc.stdout}")
    return proc.stdout


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--paper-dir",
        type=Path,
        default=Path("paper/hpca1_hail_mary/revised_paper"),
    )
    parser.add_argument("--json-out", type=Path)
    args = parser.parse_args()
    paper_dir = args.paper_dir.resolve()
    tex_path = paper_dir / "main.tex"
    pdf_path = paper_dir / "main.pdf"
    log_path = paper_dir / "main.log"
    source = tex_path.read_text(encoding="utf-8")

    checks: list[dict[str, object]] = []

    def record(name: str, passed: bool, detail: str) -> None:
        checks.append({"name": name, "passed": bool(passed), "detail": detail})

    record(
        "submission_number_source",
        r"\newcommand{\hpcasubmissionnumber}{4950}" in source,
        "main.tex must define submission number 4950 exactly",
    )
    draft_calls = len(re.findall(r"\\draftvalue\s*\{", source))
    record("no_draftvalue_calls", draft_calls == 0, f"rendered-use count={draft_calls}")
    record(
        "approved_ai_disclosure_unchanged",
        APPROVED_AI_DISCLOSURE in source,
        "approved appendix text must remain verbatim",
    )
    record("pdf_exists", pdf_path.is_file(), str(pdf_path))
    record("log_exists", log_path.is_file(), str(log_path))

    if pdf_path.is_file():
        pdf_text = run("pdftotext", str(pdf_path), "-", cwd=paper_dir)
        lowered = pdf_text.lower()
        record("submission_number_rendered", "4950" in pdf_text, "PDF contains 4950")
        prohibited = [
            token
            for token in ("author input", "placeholder", "tbd", "todo", "dummy value", "fake value", "#xxx")
            if token in lowered
        ]
        record("no_prohibited_draft_text", not prohibited, f"matches={prohibited}")
        anonymity_terms = [
            token
            for token in ("/home/rishabh", "rbsparky", "github.com/rbsparky")
            if token in lowered
        ]
        record("visible_anonymity_scan", not anonymity_terms, f"matches={anonymity_terms}")

        info = run("pdfinfo", str(pdf_path), cwd=paper_dir)
        info_map: dict[str, str] = {}
        for line in info.splitlines():
            if ":" in line:
                key, value = line.split(":", 1)
                info_map[key.strip()] = value.strip()
        pages = int(info_map.get("Pages", "-1"))
        record("page_count", pages <= 10, f"pages={pages}; current package target=10")
        page_size = info_map.get("Page size", "")
        record("us_letter", "612 x 792" in page_size, page_size)
        author = info_map.get("Author", "")
        record("blank_pdf_author", not author, f"Author={author!r}")
        qpdf_output = run("qpdf", "--check", str(pdf_path), cwd=paper_dir)
        record("qpdf_integrity", "No syntax or stream encoding errors found" in qpdf_output, qpdf_output.strip())

        fonts = run("pdffonts", str(pdf_path), cwd=paper_dir)
        font_lines = [line for line in fonts.splitlines()[2:] if line.strip()]
        unembedded = [line for line in font_lines if re.search(r"\sno\s+(?:yes|no)\s+(?:yes|no)\s+\d", line)]
        record("fonts_present", bool(font_lines), f"font_rows={len(font_lines)}")
        record("fonts_embedded", not unembedded, f"unembedded_rows={len(unembedded)}")

    if log_path.is_file():
        log = log_path.read_text(encoding="utf-8", errors="replace")
        undefined = re.findall(r"(?:undefined references|Citation .* undefined|Reference .* undefined)", log, re.I)
        overfull = re.findall(r"Overfull \\[hv]box", log)
        record("no_undefined_references", not undefined, f"matches={len(undefined)}")
        record("no_overfull_boxes", not overfull, f"matches={len(overfull)}")

    passed = all(bool(item["passed"]) for item in checks)
    report = {
        "schema": "hpca1_paper_preflight_v1",
        "paper_dir": str(paper_dir),
        "passed": passed,
        "checks": checks,
    }
    out_path = args.json_out or (paper_dir / "BUILD_PREFLIGHT.json")
    if not out_path.is_absolute():
        out_path = Path.cwd() / out_path
    atomic_json(out_path, report)
    for item in checks:
        print(f"{'PASS' if item['passed'] else 'FAIL'} {item['name']}: {item['detail']}")
    print(f"OVERALL: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
