from pydantic import BaseModel, Field
from typing import Literal


class Router(BaseModel):
    destination: Literal["rag", "direct", "code"] = Field(
        description="route for the model to inform the user"
    )
