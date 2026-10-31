"""Data structures shared across the pipeline."""
from dataclasses import asdict, dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    doc: str          # PDF file name
    doc_label: str    # e.g. "Tesla Q3 2023 Quarterly Update"
    page: int         # 1-based page number, used for citations
    type: str         # "text" | "table" | "chart"
    text: str
    section: str = ""  # page heading, e.g. "OPERATIONAL SUMMARY"

    def index_text(self) -> str:
        """Text that is embedded / BM25-indexed / reranked.

        Prepending metadata lets retrieval use the document name and section,
        which matters because the three reports share almost identical wording.
        """
        header = f"{self.doc_label} | page {self.page} | {self.section} | {self.type}"
        return f"{header}\n{self.text}"

    def citation(self) -> str:
        return f"{self.doc_label}, p.{self.page}"

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RetrievedChunk:
    chunk: Chunk
    score: float
    # Rank of this chunk in each stage (1-based), useful for debugging the funnel.
    ranks: dict = field(default_factory=dict)
