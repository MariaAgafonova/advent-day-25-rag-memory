from dataclasses import dataclass, field

@dataclass(frozen=True)
class GenerationSettings:
    temperature: float = 0.2
    max_output_tokens: int = 512
    top_k: int = 5
    max_context_tokens: int = 3000
    context_window_tokens: int = 8192

    def validate(self):
        if not 0 <= self.temperature <= 2:
            raise ValueError("Температура должна быть от 0 до 2")
        if not 1 <= self.top_k <= 50:
            raise ValueError("top-k должен быть от 1 до 50")
        if min(self.max_output_tokens, self.max_context_tokens, self.context_window_tokens) <= 0:
            raise ValueError("Бюджеты токенов должны быть положительными")
        if self.max_output_tokens >= self.context_window_tokens:
            raise ValueError("Резерв ответа должен быть меньше окна модели")

@dataclass
class RetrievedChunk:
    chunk_id: str
    source: str
    title: str
    section: list[list[str]]
    line_start: int
    line_end: int
    page_start: int | None
    page_end: int | None
    text: str
    retrieval_score: float
    rank: int
    start_char: int
    end_char: int
    document_hash: str
    source_url: str | None = None

@dataclass
class ContextItem:
    citation_id: str
    chunk_id: str
    included_text: str
    source: str
    title: str
    section: list[list[str]]
    line_start: int
    line_end: int
    page_start: int | None
    page_end: int | None
    start_char: int
    end_char: int
    document_hash: str
    token_count: int
    token_count_method: str
    truncated: bool
    source_url: str | None = None
    provenance: list[dict] = field(default_factory=list)

@dataclass
class AnswerResult:
    request_id: str
    question: str
    mode: str
    answer: str = ""
    retrieved_chunks: list[RetrievedChunk] = field(default_factory=list)
    context_items: list[ContextItem] = field(default_factory=list)
    citations: list[str] = field(default_factory=list)
    invalid_citations: list[str] = field(default_factory=list)
    retrieval_latency_ms: float = 0
    generation_latency_ms: float = 0
    total_latency_ms: float = 0
    usage: dict | None = None
    model: str = ""
    provider: str = ""
    generation_settings: dict = field(default_factory=dict)
    index_version: dict | None = None
    prompt_version: str = ""
    status: str = "pending"
    error: dict | None = None
    finish_reason: str | None = None
    messages: list[dict] = field(default_factory=list)
    context_token_count: int = 0
    token_count_method: str = ""
    warnings: list[str] = field(default_factory=list)
    provider_attempts: int = 0
    api_request: dict | None = None
    schema_version: int = 2
    run_id: str | None = None
    question_id: str | None = None
    original_query: str = ""
    rewritten_query: str = ""
    rewrite: dict = field(default_factory=dict)
    pipeline_settings: dict = field(default_factory=dict)
    candidates: list[dict] = field(default_factory=list)
    processed_chunks: list[dict] = field(default_factory=list)
    context_text: str = ""
    generation_skipped: bool = False
    generation_calls: int = 0
    stage_latency_ms: dict = field(default_factory=dict)
    counts: dict = field(default_factory=dict)
    answer_status: str | None = None
    refusal_reason: str | None = None
    structured_answer: dict | None = None
    resolved_sources: list[dict] = field(default_factory=list)
    validation: dict = field(default_factory=dict)
    evidence_attempts: list[dict] = field(default_factory=list)
    failure_clarification: str | None = None
    policy_actions: list[str] = field(default_factory=list)
