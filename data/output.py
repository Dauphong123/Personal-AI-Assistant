from pydantic import BaseModel, Field


class CodeAnalysis(BaseModel):
    explaination: str = Field(description="Question analysis")
    relevant_files: list[str] | None = Field(
        description="Relevant files or documents from the outside source"
    )
    confidence: float = Field(description="the confidence value of the analysis")


class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field(description="Age of the person")
    description: str = Field(description="extra data of the person")
