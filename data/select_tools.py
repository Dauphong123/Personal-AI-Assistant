from pydantic import BaseModel, Field


class Tools(BaseModel):
    file_name: str = Field(description="path name of the selected file")
    tool_name: str = Field(description="tool name")
    reason: str = Field(description="Reason of selection")


class RetrievalDocument(BaseModel):
    should_retrieve: bool = Field(
        description="should the retrieve document from outside source"
    )
    query: str | None = Field(None, description="the query of the search")
