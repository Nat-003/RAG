from pydantic import BaseModel, Field, model_validator

class MinimalSource(BaseModel):
    file_path: str = Field(min_length=1)
    first_character_index: int = Field(gt=0)
    last_character_index: int

    @model_validator(mode='after')
    def check_indices(self) -> 'MinimalSource':
        if self.last_character_index <= self.first_character_index:
            raise ValueError(
                f"last_character_index ({self.last_character_index}) must be "
                f"greater than first_character_index ({self.first_character_index})"
            )
        return self

class Chunk(BaseModel):
    chunk_id: int = Field(gt=0)
    text: str = Field(min_length=1)
    source: MinimalSource

class RetrivedChunk(BaseModel):
    chunk: Chunk
    score: float

class UnansweredQuestion(BaseModel):
    question_id: int = Field(gt=0)
    question: str = Field(min_length=1)

class AnsweredQuestion(UnansweredQuestion):
    sources: list[MinimalSource]
    answer: str = Field(min_length=1)

class RagDataset(BaseModel):
    rag_questions: list[AnsweredQuestion] | list[UnansweredQuestion] = Field(min_length=1)

class MinimalSearchResults(BaseModel):
    question_id: int = Field(gt=0)
    question: str = Field(min_length=1)
    retrived_sources: list[MinimalSource]

class MinimalAnswer(MinimalSearchResults):
    answer: str = Field(min_length=1)

class StudentSearchResults(BaseModel):
    search_results: list[MinimalSearchResults]
    k: int = Field(gt=0)

class StudentSearchResulsAndAnswer(BaseModel):
    search_results: list[MinimalAnswer]
    k: int = Field(gt=0)
