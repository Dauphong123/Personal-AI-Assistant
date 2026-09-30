from pydantic import BaseModel, Field


class RagOutput(BaseModel):
    response: str = Field(description="the result string of the model")
    source: list = Field(
        [], description="List of source that the model get the info from"
    )
