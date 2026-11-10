from pydantic import BaseModel, Field, model_validator
from uuid import uuid4, UUID
from typing import Literal

class MinimalSource(BaseModel):
    """Represent a source location in the original repository."""

    file_path: str = Field(min_length=1)
    first_character_index: int = Field(ge=0)
    last_character_index: int

    @model_validator(mode="after")
    def check_indices(self) -> "MinimalSource":
        """Ensure the end index is greater than the start index."""
        if self.last_character_index <= self.first_character_index:
            raise ValueError(
                f"last_character_index ({self.last_character_index}) "
                "must be greater than "
                f"first_character_index ({self.first_character_index})"
            )
        return self

class Chunk(BaseModel):
    chunk_id: int = Field(ge=0)
    text: str = Field(min_length=1)
    source: MinimalSource

class RetrievedChunk(BaseModel):
    chunk: Chunk
    score: float = Field(allow_inf_nan=False)

class UnansweredQuestion(BaseModel):
    question_id: str = Field(
    default_factory=lambda: str(uuid4()),
    min_length=1,
    )
    question: str = Field(min_length=1)

class AnsweredQuestion(UnansweredQuestion):
    sources: list[MinimalSource]
    answer: str = Field(min_length=1)

class RagDataset(BaseModel):
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]

class MinimalSearchResults(BaseModel):
    question_id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    retrieved_sources: list[MinimalSource]

class MinimalAnswer(MinimalSearchResults):
    answer: str = Field(min_length=1)

class StudentSearchResults(BaseModel):
    search_results: list[MinimalSearchResults]
    k: int = Field(ge=0)

class StudentSearchResultsAndAnswer(BaseModel):
    search_results: list[MinimalAnswer]
    k: int = Field(ge=0)

class SourceFile(BaseModel):
    file_path: str
    text: str
    file_type: Literal["python", "markdown", "text"]
