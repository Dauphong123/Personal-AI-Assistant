from typing import Literal

from pydantic import BaseModel, Field


class Router(BaseModel):
    destination: Literal["rag", "direct", "code"] = Field(
        description="route for the model to inform the user"
    )
