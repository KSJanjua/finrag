"""Rebuild docs/ARCHITECTURE.md from the .mmd diagram sources in this folder.

    python docs/architecture/build_markdown.py

The .mmd files are the editable source of truth (open them in VS Code with a
Mermaid extension, or paste into https://mermaid.live). GitHub renders the
```mermaid blocks in the generated markdown.
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "ARCHITECTURE.md"

INTRO = """# Architecture

Diagrams of the financial multimodal RAG system, derived from the code in `src/finrag/` and `scripts/`.
Each diagram's source is an editable Mermaid file in [`docs/architecture/`](architecture/); this page is
generated from them by `python docs/architecture/build_markdown.py`.

**Not implemented** (do not read these into the diagrams): authentication, a database, a web UI,
streaming responses, Docker/deployment, background job queues. Planned work is listed under
*Next steps* in the README.
"""


def title_of(src):
    first = src.splitlines()[0]
    return first.lstrip("%").strip().split("·", 1)[-1].split(":", 1)[0].strip()


def main():
    parts = [INTRO]
    for f in sorted(HERE.glob("*.mmd")):
        src = f.read_text(encoding="utf-8").strip()
        comments = [l.lstrip("%").strip() for l in src.splitlines() if l.startswith("%%")]
        parts.append(f"## {title_of(src)}\n\n{comments[0]}  \n_Source: [`architecture/{f.name}`](architecture/{f.name})_ · "
                     f"{' '.join(comments[1:])}\n\n```mermaid\n{src}\n```\n")
    OUT.write_text("\n".join(parts), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
