"""Build study notebooks from the exact production Python modules."""
from __future__ import annotations

import ast
from pathlib import Path

import nbformat as nbf


ROOT = Path(__file__).resolve().parent

EXPLANATIONS = {
    "retrieval.py": {
        "terms": "Tokenizes text and removes general stop words so retrieval focuses on meaningful terms.",
        "load_book_documents": "Extracts usable PDF pages and turns each page into a searchable Document.",
        "load_documents": "Builds the complete knowledge base from the local catalog, JSONL collections, books, and source manifest.",
        "LocalVectorStore": "Stores semantic embeddings in persistent ChromaDB and performs cosine-similarity search.",
        "vector_store": "Caches one vector-store instance so the index is reused during the application lifetime.",
        "retrieve": "Combines exact Gardiner matches, keyword overlap, metadata signals, and semantic similarity into a ranked result.",
    },
    "augmentation.py": {
        "build_context": "Converts retrieved records into bounded evidence blocks, stable source tags, citations, and confidence.",
        "prompt_messages": "Creates the provider-ready system and user messages from the evidence packet.",
    },
    "generation_safety.py": {
        "safety_check": "Blocks unsafe requests before they reach an external model and validates the input length.",
        "generate": "Runs the safety gate, prepares grounded messages, or calls the supplied provider function.",
    },
}


def source_sections(path: Path) -> list[tuple[str, str]]:
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    lines = source.splitlines(keepends=True)
    nodes = sorted(tree.body, key=lambda node: node.lineno)
    sections = []
    for index, node in enumerate(nodes):
        # Use the next top-level node as the boundary so blank lines and
        # comments remain exactly where they are in the production file.
        end = nodes[index + 1].lineno - 1 if index + 1 < len(nodes) else len(lines)
        name = getattr(node, "name", "imports and module setup")
        sections.append((name, "".join(lines[node.lineno - 1:end])))
    return sections


def build(module_name: str) -> None:
    source_path = ROOT / module_name
    notebook = nbf.v4.new_notebook()
    notebook.metadata.kernelspec = {"display_name": ".venv (3.11.9)", "language": "python", "name": "python3"}
    notebook.cells.append(nbf.v4.new_markdown_cell(
        f"# Study notebook: `{module_name}`\n\n"
        "This notebook contains the exact production code from the matching Python module. "
        "Each explanation appears immediately before the code section it explains. "
        "The code is copied line-for-line from the application module."
    ))
    explanations = EXPLANATIONS.get(module_name, {})
    for name, code in source_sections(source_path):
        explanation = explanations.get(name, "This section is part of the production implementation and supports the module's main workflow.")
        notebook.cells.append(nbf.v4.new_markdown_cell(f"## `{name}`\n\n{explanation}"))
        notebook.cells.append(nbf.v4.new_code_cell(code))
    output_names = {
        "retrieval.py": "01_retrieval.ipynb",
        "augmentation.py": "02_augmentation.ipynb",
        "generation_safety.py": "03_generation_safety.ipynb",
    }
    output_path = ROOT / output_names[module_name]
    nbf.write(notebook, output_path)


if __name__ == "__main__":
    for module in ("retrieval.py", "augmentation.py", "generation_safety.py"):
        build(module)
