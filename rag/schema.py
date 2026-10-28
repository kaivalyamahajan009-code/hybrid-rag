from dataclasses import asdict, dataclass, field


@dataclass
class Section:
    title: str
    # (paragraph text, page number) so chunks know which pages they came from
    paragraphs: list[tuple[str, int]] = field(default_factory=list)


@dataclass
class Chunk:
    id: str
    doc: str  # pdf file name
    section: str
    page_start: int
    page_end: int
    text: str
    # NER metadata, e.g. {"ORGANIZATION": ["acme corp"], "DATE": ["march 2024"]}
    entities: dict[str, list[str]] = field(default_factory=dict)

    def pages(self) -> str:
        if self.page_start == self.page_end:
            return f"p. {self.page_start}"
        return f"pp. {self.page_start}-{self.page_end}"

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "Chunk":
        return Chunk(**d)


@dataclass
class SearchResult:
    chunk: Chunk
    score: float
    vector_rank: int | None = None
    bm25_rank: int | None = None
    rerank_score: float | None = None  # sigmoid of the cross-encoder logit


@dataclass
class Answer:
    question: str
    text: str
    sources: list[SearchResult]  # [1] in the text refers to sources[0]
    cited: list[int] = field(default_factory=list)
    abstained: bool = False
    grounded: bool = True
    unsupported: list[str] = field(default_factory=list)  # answer sentences not backed by a source
